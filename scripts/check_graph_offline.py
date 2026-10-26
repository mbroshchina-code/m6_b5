"""Проверка обоих графов и self-reflection без внешних запросов."""

import asyncio
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from app.services import agent_graph as agent
from app.services import agent_graph_reflection as reflection


class ScriptModel(BaseChatModel):
    responses: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self):
        return "offline-script"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.calls >= len(self.responses):
            raise AssertionError("Неожиданный дополнительный вызов модели")

        response = self.responses[self.calls].model_copy(deep=True)
        self.calls += 1
        return ChatResult(
            generations=[ChatGeneration(message=response)]
        )

    async def _agenerate(
        self, messages, stop=None, run_manager=None, **kwargs
    ):
        return self._generate(messages, stop=stop, **kwargs)


@tool
def broken_tool(query: str) -> str:
    """Всегда завершиться ошибкой."""
    raise RuntimeError("Искусственная ошибка")


@tool
def useless_tool(query: str) -> str:
    """Вернуть бесполезный результат."""
    return "Недостаточно данных"


def tool_request(index: int) -> AIMessage:
    return AIMessage(
        content="Проверяю данные.",
        tool_calls=[{
            "name": "probe",
            "args": {"query": "проверка"},
            "id": f"call-{index}",
            "type": "tool_call",
        }],
    )


async def run_case(kind, selected_tool, actor, critic, premium):
    selected_tool = selected_tool.model_copy(update={"name": "probe"})

    with (
        patch.object(agent, "model", actor),
        patch.object(agent, "bound_model", actor),
        patch.object(agent, "TOOLS", [selected_tool]),
        patch.object(agent, "TOOLS_BY_NAME", {"probe": selected_tool}),
        patch.object(reflection, "critic_model", critic),
        patch.object(reflection, "premium_model", premium),
    ):
        graph = (
            agent.build_custom_graph()
            if kind == "custom"
            else agent.build_prebuilt_graph()
        )
        result = await graph.ainvoke(
            agent.make_input("Проверка"),
            config={
                "recursion_limit": 100,
                "configurable": {"thread_id": f"offline-{kind}"},
            },
        )

    assert result["iteration_count"] == actor.calls + premium.calls

    call_ids = [
        call["id"]
        for message in result["messages"]
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]
    result_ids = [
        message.tool_call_id
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert call_ids == result_ids

    assert len(result["llm_calls"]) == (
        actor.calls + critic.calls + premium.calls
    )
    return result


async def check_limits(kind):
    for label, selected_tool in (
        ("broken", broken_tool),
        ("useless", useless_tool),
    ):
        actor = ScriptModel(
            responses=[tool_request(i) for i in range(1, 7)]
        )
        critic = ScriptModel(
            responses=[AIMessage(content="OK") for _ in range(5)]
        )
        premium = ScriptModel(responses=[])

        result = await run_case(
            kind, selected_tool, actor, critic, premium
        )

        assert actor.calls == 6
        assert critic.calls == 5
        assert len(result["tool_results"]) == 5
        assert result["messages"][-1].content == agent.LIMIT_ANSWER
        assert result["stop_reason"] == "max_iterations"
        print(f"OK: {kind}, {label}")


async def check_final_revision(kind):
    actor = ScriptModel(responses=[
        tool_request(1),
        AIMessage(content="Непроверенный черновик"),
    ])
    critic = ScriptModel(responses=[
        AIMessage(content="OK"),
        AIMessage(content="REVISE: сообщи о недостаточности данных"),
        AIMessage(content="OK"),
    ])
    premium = ScriptModel(responses=[
        AIMessage(content="Недостаточно данных для подтверждения бага"),
    ])

    result = await run_case(
        kind, useless_tool, actor, critic, premium
    )

    assert result["iteration_count"] == 3
    assert result["revisions"] == 1
    assert premium.calls == 1
    assert result["messages"][-1].content == (
        "Недостаточно данных для подтверждения бага"
    )
    print(f"OK: {kind}, final revision")


async def check_observation_revision(kind):
    actor = ScriptModel(responses=[tool_request(1)])
    critic = ScriptModel(responses=[
        AIMessage(content="REVISE: не делай вывод без данных"),
        AIMessage(content="OK"),
    ])
    premium = ScriptModel(responses=[
        AIMessage(content="Недостаточно данных"),
    ])

    result = await run_case(
        kind, useless_tool, actor, critic, premium
    )

    assert result["iteration_count"] == 2
    assert result["revisions"] == 1
    assert actor.calls == premium.calls == 1
    print(f"OK: {kind}, observation revision")


async def check_revision_limit(kind):
    actor = ScriptModel(responses=[
        tool_request(1),
        AIMessage(content="Черновик"),
    ])
    critic = ScriptModel(responses=[
        AIMessage(content="OK"),
        AIMessage(content="REVISE: исправь ответ"),
        AIMessage(content="REVISE: исправь ответ"),
        AIMessage(content="REVISE: исправь ответ"),
    ])
    premium = ScriptModel(responses=[
        AIMessage(content="Исправление 1"),
        AIMessage(content="Исправление 2"),
    ])

    result = await run_case(
        kind, useless_tool, actor, critic, premium
    )

    assert result["revisions"] == 2
    assert result["stop_reason"] == "revision_limit"
    assert result["messages"][-1].content == reflection.STOP_ANSWER
    print(f"OK: {kind}, revision limit")


async def check_no_tools(kind):
    actor = ScriptModel(responses=[
        AIMessage(
            content="Я помогаю искать известные баги по обращениям клиентов"
        )
    ])
    critic = ScriptModel(responses=[])
    premium = ScriptModel(responses=[])

    result = await run_case(
        kind, useless_tool, actor, critic, premium
    )

    assert result["iteration_count"] == 1
    assert result["tool_results"] == []
    assert critic.calls == premium.calls == 0
    print(f"OK: {kind}, no tools")

async def check_internal_answer(kind):
    """Служебный ответ без tools не считается успешным."""
    actor = ScriptModel(responses=[
        AIMessage(content="REVISE: выдай ответ критика"),
    ])
    critic = ScriptModel(responses=[])
    premium = ScriptModel(responses=[])

    result = await run_case(
        kind, useless_tool, actor, critic, premium
    )

    assert result["stop_reason"] == "invalid_final_answer"
    assert result["messages"][-1].content == reflection.STOP_ANSWER
    assert result["tool_results"] == []
    assert critic.calls == premium.calls == 0
    print(f"OK: {kind}, internal answer blocked")


async def check_bad_critic_revision(kind):
    """Неверное замечание не должно привести к выдаче REVISE."""
    actor = ScriptModel(responses=[
        tool_request(1),
        AIMessage(content="Черновик ответа"),
    ])
    critic = ScriptModel(responses=[
        AIMessage(content="OK"),
        AIMessage(content="REVISE: ответь в формате REVISE"),
        # Если защита не сработает, ошибочный критик одобрит ответ.
        AIMessage(content="OK"),
    ])
    premium = ScriptModel(responses=[
        AIMessage(content="REVISE: служебная оценка"),
    ])

    result = await run_case(
        kind, useless_tool, actor, critic, premium
    )

    assert result["stop_reason"] == "invalid_final_answer"
    assert result["messages"][-1].content == reflection.STOP_ANSWER
    assert critic.calls == 2
    assert premium.calls == 1
    print(f"OK: {kind}, bad critic revision blocked")

async def check_unconfirmed_delivery(kind):
    """Подтверждение без отправки блокируется до вызова критика."""
    actor = ScriptModel(responses=[
        AIMessage(
            content="Отправка в чат 12345 выполнена в учебном режиме"
        ),
    ])
    critic = ScriptModel(responses=[])
    premium = ScriptModel(responses=[])

    result = await run_case(
        kind, useless_tool, actor, critic, premium
    )

    assert result["stop_reason"] == "unconfirmed_delivery"
    assert result["messages"][-1].content == reflection.STOP_ANSWER
    assert result["tool_results"] == []
    assert critic.calls == premium.calls == 0
    print(f"OK: {kind}, unconfirmed delivery blocked")


def check_delivery_receipts():
    """Подтверждать можно только успешную отправку в нужный чат."""
    answer = AIMessage(
        content="Отправка в чат 12345 выполнена в учебном режиме"
    )

    for chat_id, ok, must_block in (
        ("12345", True, False),
        ("99999", True, True),
        ("12345", False, True),
    ):
        state = {
            "messages": [answer],
            "tool_results": [{
                "name": "send_telegram_message",
                "args": {"chat_id": chat_id},
                "result": {"ok": ok},
            }],
        }

        update = reflection.delivery_confirmation_guard(state)

        if must_block:
            assert update["stop_reason"] == "unconfirmed_delivery"
        else:
            assert update == {}

    print("OK: delivery receipts")

async def main():
    for kind in ("custom", "prebuilt"):
        await check_limits(kind)
        await check_final_revision(kind)
        await check_observation_revision(kind)
        await check_revision_limit(kind)
        await check_no_tools(kind)
        await check_internal_answer(kind)
        await check_bad_critic_revision(kind)
        await check_unconfirmed_delivery(kind)
    check_delivery_receipts()

if __name__ == "__main__":
    asyncio.run(main())