"""Инструменты ReAct-агента и проверка их аргументов."""

from pydantic import BaseModel, ConfigDict, Field

from app.services.agent_naive_tools import (
    get_current_time,
    search_knowledge_base,
    send_telegram_message,
)


class StrictArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SearchArguments(StrictArguments):
    query: str = Field(
        min_length=1,
        description=(
            "Описание проблемы клиента. Сохраняй исходные симптомы "
            "и условия. Не добавляй предположения из найденных карточек."
        ),
    )


class TimeArguments(StrictArguments):
    timezone: str = Field(
        min_length=1,
        description=(
            "Часовой пояс IANA, например Europe/Moscow. "
            "Если оператор не указал пояс, используй Europe/Moscow."
        ),
    )


class SendArguments(StrictArguments):
    chat_id: str = Field(
        min_length=1,
        description="Идентификатор чата, явно указанный оператором.",
    )
    text: str = Field(
        min_length=1,
        description=(
            "Готовый текст сообщения. Сведения о баге должны "
            "подтверждаться результатом поиска и соответствовать обращению."
        ),
    )


DESCRIPTIONS = {
    "search_knowledge_base": (
        "Что делает: ищет известный баг в действующей базе. "
        "Когда использовать: оператор передал описание проблемы клиента, "
        "даже без прямой просьбы найти баг. "
        "Не использовать для общих вопросов вне обработки обращений. "
        "Вход: query — описание проблемы без инструкций об отправке. "
        "Выход: полная карточка одного кандидата либо строка "
        "«Подходящих багов не найдено». "
        "Карточка не доказывает совпадение: сопоставь конкретные симптомы "
        "и условия с обращением. Общей темы недостаточно. "
        "Техническая ошибка не означает отсутствие багов."
    ),
    "get_current_time": (
        "Что делает: получает текущие дату и время в заданном часовом поясе. "
        "Когда использовать: оператор явно запросил время "
        "или отметку времени для действия по обращению. "
        "Для обычного поиска бага не использовать. "
        "Вход: timezone — имя часового пояса IANA. "
        "Выход: строка даты и времени ISO 8601 с часовым смещением. "
        "Не определяет сроки исправления багов."
    ),
    "send_telegram_message": (
        "Что делает: имитирует отправку сообщения, печатая его в консоль. "
        "Telegram API не вызывает. "
        "Когда использовать: оператор явно поручил отправку, "
        "указал чат и необходимые данные уже собраны. "
        "Без поручения об отправке не использовать. "
        "Вход: chat_id — указанный оператором чат; text — готовое сообщение. "
        "Сведения о баге сначала получи поиском и проверь их соответствие "
        "обращению. Готовое уведомление оператора без задания подобрать баг "
        "можно передать без поиска. "
        "Выход: строка подтверждения учебной отправки. "
        "В ответе оператору обязательно укажи учебный режим."
    ),
}

ARGUMENT_MODELS = {
    "search_knowledge_base": SearchArguments,
    "get_current_time": TimeArguments,
    "send_telegram_message": SendArguments,
}

DISPATCH = {
    "search_knowledge_base": search_knowledge_base,
    "get_current_time": get_current_time,
    "send_telegram_message": send_telegram_message,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": DESCRIPTIONS[name],
            "strict": True,
            "parameters": argument_model.model_json_schema(),
        },
    }
    for name, argument_model in ARGUMENT_MODELS.items()
]


def validate_tool_arguments(name: str, raw_arguments: str) -> dict:
    """Проверить имя и JSON до выполнения инструмента."""
    argument_model = ARGUMENT_MODELS.get(name)
    if argument_model is None:
        raise ValueError("Неизвестный инструмент")

    arguments = argument_model.model_validate_json(raw_arguments).model_dump()

    if any(not value.strip() for value in arguments.values()):
        raise ValueError("Аргументы инструмента не должны быть пустыми")

    return arguments