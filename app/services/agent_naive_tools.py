import atexit
from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

from openai import DefaultHttpxClient, OpenAI

from app.core.config import get_settings
from app.core.rag_prompts import FALLBACK_ANSWER

import json
from filelock import FileLock

def make_client():
    settings = get_settings().llm
    return OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.base_url,
        http_client=DefaultHttpxClient(
            proxy=settings.openai_proxy_url or None,
        ),
        timeout=settings.request_timeout,
        max_retries=0,
    )


@lru_cache(maxsize=1)
def get_rag():
    from app.services.rag import CitationRAGService

    service = CitationRAGService()
    service.build()
    atexit.register(service.close)
    return service


def search_knowledge_base(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Нужен непустой query")

    service = get_rag()
    with service._lock:
        with FileLock(str(service.state_dir / "ingest.lock"), timeout=5):
            candidates = service._retrieve(query)
            if not candidates:
                return FALLBACK_ANSWER

            best = max(candidates, key=lambda item: item["score"])
            if best["score"] < service.settings.rag_score_threshold:
                return FALLBACK_ANSWER

            documents = service._read_documents([best])
            document = documents[best["document_id"]]
            units = json.loads(document.text)
            text = "\n\n".join(
                unit["text"].strip()
                for unit in units
                if unit["text"].strip()
            )
            if not text:
                raise RuntimeError("Найденная карточка пуста")
            return text


def get_current_time(timezone: str = "Europe/Moscow") -> str:
    return datetime.now(ZoneInfo(timezone)).isoformat()


def send_telegram_message(chat_id: str, text: str) -> str:
    print(f"[TELEGRAM → {chat_id}] {text}")
    return f"Сообщение отправлено в {chat_id}"

DISPATCH = {
    "search_knowledge_base": search_knowledge_base,
    "get_current_time": get_current_time,
    "send_telegram_message": send_telegram_message,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Ищет известные баги в действующей базе и возвращает "
                "полную карточку одного бага, выбранного по лучшему фрагменту. "
                "Вызывай при получении описания проблемы клиента, "
                "даже без прямой просьбы найти баг. "
                "Найденная карточка является кандидатом: проверь совпадение "
                "симптомов и условий с исходным обращением перед ответом."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Описание проблемы клиента на русском языке. "
                            "Сохраняй исходные симптомы и условия; "
                            "не добавляй предположения из найденных карточек."
                        ),
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Возвращает текущие дату и время в указанном часовом "
                "поясе в формате ISO 8601. "
                "Вызывай, когда оператору нужна отметка времени "
                "для действия по обращению. "
                "Для обычного поиска бага этот инструмент не требуется."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": (
                            "Имя часового пояса IANA, "
                            "например Europe/Moscow или Asia/Vladivostok."
                        ),
                        "default": "Europe/Moscow",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_telegram_message",
            "description": (
                "Имитирует отправку сообщения клиенту: печатает текст "
                "в консоль и возвращает подтверждение; Telegram API не вызывает. "
                "Вызывай по поручению оператора, когда текст сообщения "
                "подготовлен и все необходимые для него данные собраны. "
                "Если сообщение содержит сведения о баге, сначала получи "
                "их через поиск и проверь соответствие обращению."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "string",
                        "description": "Идентификатор чата, указанный оператором.",
                    },
                    "text": {
                        "type": "string",
                        "description": (
                            "Готовый текст для клиента. "
                            "Сведения о багах должны подтверждаться поиском; "
                            "отсутствующие факты нельзя придумывать."
                        ),
                    },
                },
                "required": ["chat_id", "text"],
                "additionalProperties": False,
            },
        },
    },
]