"""Создание LangChain-модели с настройками подключения проекта."""

from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.services.agent_react import MAIN_MODEL


def make_graph_model(
    model_name: str = MAIN_MODEL,
    *,
    max_completion_tokens: int = 1800,
) -> ChatOpenAI:
    """Создать модель без выполнения запроса к API."""
    settings = get_settings().llm

    return ChatOpenAI(
        model=model_name,
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.base_url,
        openai_proxy=settings.openai_proxy_url or None,
        timeout=settings.request_timeout,
        max_retries=0,
        temperature=None,
        reasoning_effort="none",
        max_completion_tokens=max_completion_tokens,
        use_responses_api=False,
    )