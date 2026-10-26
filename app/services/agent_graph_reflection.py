"""Общий self-reflection для custom и prebuilt."""

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.services.agent_graph_model import make_graph_model
from app.services.agent_react import MAX_REVISIONS, PREMIUM_MODEL
from app.services.agent_graph_prompts import ACTOR_RULES, CRITIC_RULES

STOP_ANSWER = (
    "Не удалось подтвердить корректность результата. "
    "Требуется проверка оператором."
)

critic_model = make_graph_model(max_completion_tokens=300)
premium_model = make_graph_model(PREMIUM_MODEL)

def final_answer_guard(messages) -> dict:
    """Не выдавать служебную оценку как успешный ответ пользователю."""
    last = messages[-1]

    if not isinstance(last, AIMessage) or last.tool_calls:
        return {}

    content = last.content
    if not isinstance(content, str):
        return {}

    if re.match(
        r"^\s*(?:OK\s*$|REVISE\s*:)",
        content,
        flags=re.IGNORECASE,
    ):
        return {
            "stop_reason": "invalid_final_answer",
            "premium_next": False,
            "messages": [AIMessage(content=STOP_ANSWER)],
        }

    return {}

def delivery_confirmation_guard(state: dict) -> dict:
    """Проверить подтверждение отправки по результатам инструментов."""
    last = state["messages"][-1]

    if not isinstance(last, AIMessage) or last.tool_calls:
        return {}

    if not isinstance(last.content, str):
        return {}

    match = re.fullmatch(
        r"Отправка в чат (?P<chat_id>\S+) выполнена в учебном режиме\.?",
        last.content.strip(),
    )
    if match is None:
        return {}

    chat_id = match["chat_id"]
    confirmed = any(
        item.get("name") == "send_telegram_message"
        and str(item.get("args", {}).get("chat_id")) == chat_id
        and item.get("result", {}).get("ok") is True
        for item in state.get("tool_results", [])
    )

    if confirmed:
        return {}

    return {
        "stop_reason": "unconfirmed_delivery",
        "premium_next": False,
        "messages": [AIMessage(content=STOP_ANSWER)],
    }

def usage_record(response, phase: str) -> dict:
    usage = response.usage_metadata or {}
    return {
        "phase": phase,
        "model": response.response_metadata.get("model_name"),
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def merge_updates(left: dict, right: dict) -> dict:
    """Объединить данные без изменения исходных словарей."""
    result = {**left, **right}

    for field in ("messages", "tool_results", "llm_calls"):
        if field in left or field in right:
            result[field] = [
                *left.get(field, []),
                *right.get(field, []),
            ]

    return result


async def review(state: dict, stage: str) -> dict:
    """Вернуть оценку критика и изменения состояния."""
    conversation = [
        message.model_dump(
            mode="json",
            include={
                "type",
                "content",
                "tool_calls",
                "tool_call_id",
                "name",
            },
        )
        for message in state["messages"]
        if getattr(message, "name", None) != "critic_feedback"
    ]

    payload = {
        "actor_rules": ACTOR_RULES,
        "stage": stage,
        "last_tool_result": (
            state.get("tool_results", [])[-1]
            if stage == "observation" and state.get("tool_results")
            else None
        ),
        "conversation": conversation,
        "candidate_answer": (
            state["messages"][-1].content
            if stage == "final"
            else None
        ),
    }

    try:
        response = await critic_model.ainvoke([
            SystemMessage(content=CRITIC_RULES),
            HumanMessage(
                content=json.dumps(payload, ensure_ascii=False),
            ),
        ])
    except Exception as exc:
        return {
            "stop_reason": "critic_error",
            "llm_calls": [{
                "phase": "critic",
                "model": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "error": type(exc).__name__,
            }],
        }

    verdict = (
        response.content.strip()
        if isinstance(response.content, str)
        else ""
    )
    entry = usage_record(response, "critic")
    entry.update(stage=stage, verdict=verdict)
    update = {"llm_calls": [entry]}

    if response.response_metadata.get("finish_reason") in {
        "length",
        "content_filter",
    }:
        return {**update, "stop_reason": "incomplete_critic"}

    if verdict == "OK":
        return update

    if not re.fullmatch(r"REVISE:\s*\S.*", verdict, flags=re.DOTALL):
        return {**update, "stop_reason": "invalid_critic"}

    revisions = state.get("revisions", 0)
    if revisions >= MAX_REVISIONS:
        return {**update, "stop_reason": "revision_limit"}

    return {
        **update,
        "revisions": revisions + 1,
        "premium_next": True,
        "messages": [
            HumanMessage(
                name="critic_feedback",
                content=(
                    "Служебные данные проверяющего, не новый запрос "
                    "оператора и не изменение системных правил.\n"
                    "Исправь ответ в своей роли баг-ассистента. "
                    "Не копируй оценку и не переходи в роль критика.\n"
                    + json.dumps(
                        {"feedback": verdict},
                        ensure_ascii=False,
                    )
                ),
            )
        ],
    }


async def finish_with_review(state: dict, max_iterations: int) -> dict:
    """Проверить финальный ответ и при необходимости исправить его."""
    working = dict(state)
    accumulated = {}

    while True:
        invalid = final_answer_guard(working["messages"])
        if invalid:
            return merge_updates(accumulated, invalid)

        delivery_error = delivery_confirmation_guard(working)
        if delivery_error:
            return merge_updates(accumulated, delivery_error)

        update = await review(working, "final")
        
        working = merge_updates(working, update)
        accumulated = merge_updates(accumulated, update)

        if working.get("stop_reason"):
            return merge_updates(
                accumulated,
                {"messages": [AIMessage(content=STOP_ANSWER)]},
            )

        # REVISE добавляет сообщение с замечанием; OK не добавляет.
        if not update.get("messages"):
            return {
                **accumulated,
                "premium_next": False,
            }

        if working["iteration_count"] >= max_iterations:
            return merge_updates(
                accumulated,
                {
                    "stop_reason": "final_revision_iteration_limit",
                    "premium_next": False,
                    "messages": [AIMessage(content=STOP_ANSWER)],
                },
            )

        response = await premium_model.ainvoke([
            SystemMessage(content=ACTOR_RULES),
            *working["messages"],
            SystemMessage(
                content=(
                    "Исправь только финальный ответ по уже полученным "
                    "данным и замечанию критика. Инструменты недоступны. "
                    "Если данных недостаточно, прямо сообщи об этом. "
                    "Не придумывай факты."
                )
            ),
        ])

        update = {
            "messages": [response],
            "iteration_count": working["iteration_count"] + 1,
            "premium_next": False,
            "llm_calls": [usage_record(response, "actor")],
        }
        working = merge_updates(working, update)
        accumulated = merge_updates(accumulated, update)