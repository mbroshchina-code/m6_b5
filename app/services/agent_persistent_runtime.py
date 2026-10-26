"""Рабочие узлы ReAct для persistent-графа."""

import json

from filelock import FileLock
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.services.agent_persistent_cards import card_from_text, relevant_ids

def make_nodes(rag):
    # Офлайн-демо передаёт свои узлы и сюда не заходит.
    from app.services import agent_graph as original
    from app.services.agent_react_tools import SearchArguments
    from app.core.rag_prompts import FALLBACK_ANSWER

    reflection = original.reflection

    # Основной промпт и правила поиска сохраняются.
    # Учебная отправка из бенчмарка новому графу не нужна.
    actor_rules = original.SYSTEM_PROMPT.partition(
        "\nУЧЕБНОЕ РАСШИРЕНИЕ ДЛЯ БЕНЧМАРКА:"
    )[0]

    @tool(
        args_schema=SearchArguments,
        description=(
            "Ищет кандидатов в известные баги по обращению клиента. "
            "Используй историю для понимания уточнений. "
            "Не вызывай для сторонних вопросов. "
            "Возвращает полные карточки нескольких кандидатов. "
            "Сам факт нахождения карточки не доказывает релевантность: "
            "сопоставь симптомы и условия с обращением."
        ),
    )
    def search_knowledge_base(query: str) -> list[dict]:
        """Получить карточки из действующей базы приложения."""
        if rag is None:
            raise RuntimeError("RAG недоступен")
        if not query.strip():
            raise ValueError("Нужен непустой query")

        with rag._lock:
            with FileLock(str(rag.state_dir / "ingest.lock"), timeout=5):
                candidates = [
                    item
                    for item in rag._retrieve(query)
                    if item["score"] >= rag.settings.rag_score_threshold
                ]
                if not candidates:
                    return []

                documents = rag._read_documents(candidates)

            if rag.settings.rag_reranker_enabled:
                if rag._reranker is None:
                    from app.services.reranker import Reranker

                    rag._reranker = Reranker()

                candidates = rag._reranker.rerank(
                    query,
                    candidates,
                    top_n=rag.settings.rag_context_top_n,
                )
            else:
                candidates = candidates[:rag.settings.rag_context_top_n]

            cards = []
            for candidate in candidates:
                document = documents[candidate["document_id"]]
                units = json.loads(document.text)
                text = "\n\n".join(
                    unit["text"].strip()
                    for unit in units
                    if unit["text"].strip()
                )
                if not text:
                    raise RuntimeError("Найденная карточка пуста")

                cards.append(card_from_text(text, document.metadata))

            return cards

    tools = [search_knowledge_base]
    regular = original.model.bind_tools(
        tools,
        strict=True,
        parallel_tool_calls=False,
    )
    premium = reflection.premium_model.bind_tools(
        tools,
        strict=True,
        parallel_tool_calls=False,
    )

    async def call_model(state, config: RunnableConfig):
        if state.get("stop_reason"):
            return {
                "messages": [AIMessage(content=reflection.STOP_ANSWER)],
                "premium_next": False,
            }

        use_premium = state.get("premium_next", False)
        selected = premium if use_premium else regular

        model_messages = [
            SystemMessage(content=actor_rules),
            *state["messages"],
        ]

        response = await selected.ainvoke(
            model_messages,
            config=config,
        )
        usage = [reflection.usage_record(response, "actor")]

        searched_this_turn = any(
            item.get("name") == "search_knowledge_base"
            for item in state.get("tool_results", [])
        )

        answer_without_search = (
            not response.tool_calls
            and isinstance(response.content, str)
            and bool(relevant_ids(response.content))
            and not searched_this_turn
        )

        if answer_without_search:
            # Ответ из истории не заменяет исходные карточки.
            # Модель составляет поисковый запрос с учётом всей истории.
            base_model = (
                reflection.premium_model
                if use_premium
                else original.model
            )
            search_required = base_model.bind_tools(
                tools,
                tool_choice="search_knowledge_base",
                strict=True,
                parallel_tool_calls=False,
            )

            response = await search_required.ainvoke(
                model_messages,
                config=config,
            )
            usage.append(reflection.usage_record(response, "actor"))

        return {
            "messages": [response],
            "iteration_count": state["iteration_count"] + 1,
            "premium_next": False,
            "llm_calls": usage,
        }

    async def execute_tool(state, config: RunnableConfig):
        messages = []
        results = []
        source_cards = []

        for call in state["messages"][-1].tool_calls:
            if call["name"] != "search_knowledge_base":
                outcome = {"ok": False, "error": "Инструмент недоступен"}
            else:
                try:
                    cards = await search_knowledge_base.ainvoke(
                        call["args"],
                        config=config,
                    )
                    source_cards.extend(cards)

                    value = (
                        json.dumps(
                            [
                                {"bug_id": card["bug_id"], "text": card["text"]}
                                for card in cards
                            ],
                            ensure_ascii=False,
                        )
                        if cards
                        else FALLBACK_ANSWER
                    )
                    outcome = {"ok": True, "value": value}
                except Exception as exc:
                    outcome = {"ok": False, "error": type(exc).__name__}

            messages.append(
                ToolMessage(
                    content=json.dumps(outcome, ensure_ascii=False),
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="success" if outcome["ok"] else "error",
                )
            )
            results.append(
                {
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "args": call["args"],
                    "result": outcome,
                }
            )

        update = {
            "messages": messages,
            "tool_results": results,
            "source_cards": source_cards,
        }
        working = reflection.merge_updates(state, update)
        assessment = await reflection.review(working, "observation")
        return reflection.merge_updates(update, assessment)

    return {
        "call_model": call_model,
        "execute_tool": execute_tool,
        "force_finish": original.force_finish,
    }