"""Общие инструменты для custom и prebuilt LangGraph-агентов."""

import json

from langchain_core.tools import tool

from app.services.agent_react_tools import (
    DESCRIPTIONS,
    DISPATCH,
    SearchArguments,
    SendArguments,
    TimeArguments,
    validate_tool_arguments,
)


def _execute(name: str, arguments: dict) -> str:
    """Проверить аргументы и вызвать существующий инструмент Б6.2."""
    validated = validate_tool_arguments(
        name,
        json.dumps(arguments, ensure_ascii=False),
    )
    return str(DISPATCH[name](**validated))


@tool(
    args_schema=SearchArguments,
    description=DESCRIPTIONS["search_knowledge_base"],
)
def search_knowledge_base(query: str) -> str:
    """Найти кандидата в известные баги по обращению клиента."""
    return _execute(
        "search_knowledge_base",
        {"query": query},
    )


@tool(
    args_schema=TimeArguments,
    description=DESCRIPTIONS["get_current_time"],
)
def get_current_time(timezone: str) -> str:
    """Получить отметку времени для обработки обращения."""
    return _execute(
        "get_current_time",
        {"timezone": timezone},
    )


@tool(
    args_schema=SendArguments,
    description=DESCRIPTIONS["send_telegram_message"],
)
def send_telegram_message(chat_id: str, text: str) -> str:
    """Выполнить учебную отправку подготовленного сообщения."""
    return _execute(
        "send_telegram_message",
        {"chat_id": chat_id, "text": text},
    )


TOOLS = [
    search_knowledge_base,
    get_current_time,
    send_telegram_message,
]