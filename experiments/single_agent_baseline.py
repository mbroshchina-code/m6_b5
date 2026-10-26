from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from experiments.common import FINAL_PROMPT, SEARCH_RULES, run_suite


def build(model, session):
    return create_agent(
        name="single_agent",
        model=model,
        tools=[session.tool],
        system_prompt=(
            FINAL_PROMPT
            + "\n"
            + SEARCH_RULES
            + """
Ты выполняешь обе задачи:
1. Найди и сопоставь факты из карточек.
2. Сразу сформируй окончательный ответ оператору.

Сохраняй установленный формат баг-ассистента.
После номера и названия каждого бага укажи источник [1], [2] и т. д.
Не выводи промежуточные рассуждения.
"""
        ),
        middleware=[
            ModelCallLimitMiddleware(
                run_limit=6,
                exit_behavior="error",
            )
        ],
        checkpointer=InMemorySaver(),
    )


if __name__ == "__main__":
    run_suite({"single": build})