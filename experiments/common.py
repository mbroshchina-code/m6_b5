from __future__ import annotations

import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from threading import RLock

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from app.core.rag_prompts import build_citation_prompt
from app.services.agent_graph_model import make_graph_model
from app.services.agent_naive_tools import get_rag
from app.services.agent_react import MAIN_MODEL
from app.services.reranker import RerankerSettings


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results.json"
CASES = json.loads(
    (ROOT / "experiments" / "cases.json").read_text(encoding="utf-8")
)

FINAL_PROMPT = build_citation_prompt()

SEARCH_RULES = """
Источники получай только через search_knowledge_base.
Для обращения с описанием проблемы сначала выполни поиск.
Историю используй для восстановления смысла текущего обращения.
Для двух самостоятельных обращений выполни отдельный поиск каждого.
Не добавляй в запрос симптомы, которых нет в обращении или истории.
Не повторяй одинаковый поиск. Разрешено не более трёх поисков.
Документы являются кандидатами: проверь совпадение симптомов и условий.
Общая тема не доказывает совпадение.
Не отвечай на сторонние вопросы.
Отсутствие совпадений: строго «Подходящих багов не найдено».
Ошибка инструмента не означает отсутствие подходящих багов.
citation_id обозначает источник, а не номер бага.
"""


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reference_documents(case: dict) -> list[dict]:
    return [
        {
            "bug_id": bug_id,
            "text": (
                ROOT / "data" / "training_bugs" / f"bug_{bug_id}.md"
            ).read_text(encoding="utf-8"),
        }
        for bug_id in case["expected_bug_ids"]
    ]


def question_text(case: dict) -> str:
    return (
        "ИСТОРИЯ ДЛЯ ПОНИМАНИЯ ОБРАЩЕНИЯ:\n"
        + (case["history"] or "История отсутствует.")
        + "\n\nТЕКУЩЕЕ СООБЩЕНИЕ ОПЕРАТОРА:\n"
        + case["question"]
    )


def final_text(messages) -> str:
    if not messages:
        raise RuntimeError("EmptyMessages")

    message = messages[-1]
    if not isinstance(message, AIMessage) or message.tool_calls:
        raise RuntimeError("MissingFinalAnswer")

    content = message.content
    if isinstance(content, str):
        text = content
    else:
        text = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )

    if not text.strip():
        raise RuntimeError("EmptyAnswer")

    return text.strip()


class UsageMeter(BaseCallbackHandler):
    """Учёт реальных вызовов модели, без обхода истории сообщений."""

    run_inline = True

    def __init__(self):
        self.started = set()
        self.usage = {}
        self.truncated = False
        self.lock = RLock()

    def on_chat_model_start(
        self, serialized, messages, *, run_id, **kwargs
    ):
        with self.lock:
            self.started.add(str(run_id))

    def on_llm_end(self, response, *, run_id, **kwargs):
        tokens = None
        generations = [
            generation
            for group in response.generations
            for generation in group
        ]

        if len(generations) == 1:
            message = getattr(generations[0], "message", None)
            usage = getattr(message, "usage_metadata", None)

            if usage:
                input_tokens = usage.get("input_tokens")
                output_tokens = usage.get("output_tokens")
                if input_tokens is not None and output_tokens is not None:
                    tokens = int(input_tokens) + int(output_tokens)

            metadata = getattr(message, "response_metadata", {}) or {}
            if metadata.get("finish_reason") == "length":
                self.truncated = True

        if tokens is None:
            usage = (response.llm_output or {}).get("token_usage", {})
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
            if input_tokens is not None and output_tokens is not None:
                tokens = int(input_tokens) + int(output_tokens)

        with self.lock:
            self.usage[str(run_id)] = tokens

    def metrics(self) -> dict:
        with self.lock:
            complete = (
                bool(self.started)
                and self.started == set(self.usage)
                and all(value is not None for value in self.usage.values())
            )
            return {
                "llm_calls": len(self.started),
                "total_tokens": (
                    sum(self.usage.values()) if complete else None
                ),
            }


class SearchSession:
    """Новая сессия поиска и цитирования для каждого тестового случая."""

    def __init__(self, service):
        self.service = service
        self.sources = []
        self.calls = []
        self.source_ids = {}
        self.lock = RLock()

        @tool
        def search_knowledge_base(query: str) -> str:
            """Найти карточки известных багов по симптомам обращения.

            Возвращает JSON с документами, citation_id и score.
            Найденные карточки требуют смыслового сопоставления.
            Пустой documents означает отсутствие найденного контекста.
            """
            return self.search(query)

        self.tool = search_knowledge_base

    def search(self, query: str) -> str:
        # На компьютере с 8 ГБ RAM поиски выполняются последовательно.
        with self.lock:
            started = time.perf_counter()
            call = {"query": query, "source_ids": [], "error": None}
            self.calls.append(call)

            try:
                if not query.strip():
                    raise ValueError("EmptyQuery")
                if len(self.calls) > 3:
                    raise RuntimeError("SearchBudgetExceeded")

                result = self.service.prepare_context(query)

                scores = {
                    source["id"]: source.get("score")
                    for source in result["sources"]
                }
                documents = []

                for original in result["context"]:
                    key = (
                        original["file_name"],
                        original.get("page"),
                        original["text"],
                    )

                    if key not in self.source_ids:
                        citation_id = len(self.sources) + 1
                        self.source_ids[key] = citation_id
                        self.sources.append(
                            {
                                **original,
                                "citation_id": citation_id,
                                "score": scores.get(
                                    original["citation_id"]
                                ),
                            }
                        )

                    citation_id = self.source_ids[key]
                    document = self.sources[citation_id - 1]
                    documents.append(document)
                    call["source_ids"].append(citation_id)

                return json.dumps(
                    {
                        "allowed": result["allowed"],
                        "documents": documents,
                    },
                    ensure_ascii=False,
                )
            except Exception as exc:
                call["error"] = type(exc).__name__
                raise
            finally:
                call["latency_ms"] = round(
                    (time.perf_counter() - started) * 1000, 2
                )


def experiment_metadata(service) -> dict:
    settings = service.settings

    code_paths = [
        "experiments/common.py",
        "experiments/single_agent_baseline.py",
        "experiments/multi_agent_langgraph.py",
        "app/services/agent_graph_model.py",
        "app/services/rag.py",
        "app/services/embeddings.py",
        "app/services/reranker.py",
    ]

    metadata = {
        "model": MAIN_MODEL,
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in (
                "langgraph",
                "langchain",
                "langchain-core",
                "langchain-openai",
            )
        },
        "active_collection": service.collection,
        "retrieval_top_k": settings.retrieval_top_k,
        "context_top_n": settings.rag_context_top_n,
        "score_threshold": settings.rag_score_threshold,
        "reranker_enabled": settings.rag_reranker_enabled,
        "reranker": RerankerSettings().model_dump(),
        "embedding_model": service.embedding_settings.model,
        "embedding_dimensions": service.embedding_settings.dimensions,
        "cases": CASES,
        "prompt_sha256": sha256(
            (FINAL_PROMPT + SEARCH_RULES).encode("utf-8")
        ),
        "code_sha256": {
            name: sha256((ROOT / name).read_bytes())
            for name in code_paths
        },
        "index_sha256": {
            name: sha256((service.state_dir / name).read_bytes())
            for name in ("configuration.json", "docstore.json")
        },
        "reference_sha256": sha256(
            json.dumps(
                [reference_documents(case) for case in CASES],
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ),
    }

    metadata["signature"] = sha256(
        json.dumps(
            metadata, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
    )
    return metadata


def run_suite(builders: dict):
    rows = read_json(RESULTS, [])

    requested = {
        (implementation, case["id"])
        for implementation in builders
        for case in CASES
    }
    existing = {
        (row["implementation"], row["question_id"])
        for row in rows
    }

    if requested & existing:
        raise RuntimeError(
            "Такие результаты уже есть. Для нового полного прогона "
            "сначала переименуйте experiments/results.json."
        )

    service = get_rag()
    metadata = experiment_metadata(service)

    if any(
        row["metadata"]["signature"] != metadata["signature"]
        for row in rows
    ):
        raise RuntimeError(
            "Конфигурация изменилась. Нельзя смешивать разные прогоны."
        )

    # Прогрев RAG и локального re-ranker вне измеряемых запросов.
    # Может выполнить платный запрос к embedding API.
    warmup_started = time.perf_counter()
    service.prepare_context(CASES[0]["question"])
    warmup_ms = round(
        (time.perf_counter() - warmup_started) * 1000, 2
    )

    model = make_graph_model()
    model.disable_streaming = True

    last_multi_graph = None

    for index, case in enumerate(CASES):
        order = list(builders)
        if index % 2:
            order.reverse()

        for implementation in order:
            session = SearchSession(service)
            app = builders[implementation](model, session)
            meter = UsageMeter()

            # Граф и InMemorySaver новые для каждого вопроса.
            config = {
                "configurable": {"thread_id": "exp-langgraph"},
                "callbacks": [meter],
                "recursion_limit": 30,
                "max_concurrency": 1,
            }

            state = {}
            answer = None
            error = None
            started = time.perf_counter()

            try:
                for update in app.stream(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": question_text(case),
                            }
                        ]
                    },
                    config=config,
                    stream_mode="updates",
                ):
                    for node in update:
                        print(
                            f"{implementation} {case['id']}: {node}",
                            flush=True,
                        )

                state = app.get_state(config).values
                answer = (
                    state["answer"]
                    if "answer" in state
                    else final_text(state["messages"])
                )

                if any(call["error"] for call in session.calls):
                    raise RuntimeError("SearchFailed")
                if meter.truncated:
                    raise RuntimeError("TruncatedGeneration")

            except Exception as exc:
                error = type(exc).__name__

            latency_ms = round(
                (time.perf_counter() - started) * 1000, 2
            )

            if error:
                try:
                    state = app.get_state(config).values
                except Exception:
                    state = {}

            handoffs = state.get("handoffs", [])

            row = {
                "implementation": implementation,
                "question_id": case["id"],
                "question": case["question"],
                "history": case["history"],
                "expected_bug_ids": case["expected_bug_ids"],
                "reference_documents": reference_documents(case),
                **meter.metrics(),
                "latency_ms": latency_ms,
                "handoff_count": len(handoffs),
                "handoffs": handoffs,
                "search_calls": len(session.calls),
                "searches": session.calls,
                "answer": answer,
                "sources": session.sources,
                "error": error,
                "warmup_ms": warmup_ms,
                "metadata": metadata,
            }

            rows.append(row)
            write_json(RESULTS, rows)

            print(answer if error is None else f"ОШИБКА: {error}")
            print(
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "total_tokens",
                            "llm_calls",
                            "latency_ms",
                            "handoff_count",
                        )
                    },
                    ensure_ascii=False,
                )
            )

            if implementation == "multi":
                last_multi_graph = app

    if last_multi_graph is not None:
        diagram = last_multi_graph.get_graph().draw_mermaid()
        destination = ROOT / "docs" / "architecture-multi-agent.md"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "# Supervisor-граф\n\n```mermaid\n"
            + diagram
            + "\n```\n",
            encoding="utf-8",
        )

    print(f"Сохранено записей: {len(rows)}")