"""Офлайн: выбор решения, interrupt, история, replay, approve/reject."""

import asyncio
import copy
import getpass
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.services.agent_persistent import (
    APPROVE_NODE,
    CHOOSE_NODE,
    build_agent,
)
from app.services.agent_persistent_cards import card_from_text


SOLUTION_ONE = "Выполнить демонстрационное действие А."
SOLUTION_TWO = "Выполнить демонстрационное действие Б.\nСохранить результат."


def demo_card(number, solution):
    text = (
        f"## Номер бага\n\n{number}\n\n"
        f"## Название\n\nДемонстрационный баг {number}\n\n"
        f"## Временное решение\n\n{solution}"
    )
    return card_from_text(
        text,
        {
            "bug_id": number,
            "file_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        },
    )


DEMO_CARDS = [
    demo_card(900001, SOLUTION_ONE),
    demo_card(900002, SOLUTION_TWO),
    demo_card(900003, "Решение менее релевантного бага."),
]

DEMO_ANSWER = (
    "Релевантные баги\n"
    "* **900001** — Демонстрационный баг 900001\n"
    "    * *Статус:* Не указано\n"
    "    * *Причина:* Не указано\n"
    f"    * *Временное решение:* {SOLUTION_ONE}\n"
    "* **900002** — Демонстрационный баг 900002\n"
    "    * *Статус:* Не указано\n"
    "    * *Причина:* Не указано\n"
    "    * *Временное решение:* Краткий пересказ модели для теста\n\n"
    "Менее релевантные баги\n"
    "* **900003** — Демонстрационный баг 900003\n"
    "    * *Статус:* Не указано\n"
    "    * *Причина:* Не указано\n"
    "    * *Временное решение:* Решение менее релевантного бага."
)

DEMO_INPUT = {
    "request_id": "offline-case-001",
    "messages": [{"role": "user", "content": "Демонстрационное обращение"}],
    "send_to": "123456789",
}


def demo_nodes(answer=DEMO_ANSWER, cards=None):
    selected_cards = DEMO_CARDS if cards is None else cards

    async def call_model(state):
        return {
            "messages": [AIMessage(content=answer)],
            "iteration_count": state["iteration_count"] + 1,
            "source_cards": copy.deepcopy(selected_cards),
        }

    async def execute_tool(state):
        raise AssertionError("Офлайн-модель не вызывает поиск")

    async def force_finish(state):
        return {}

    return {
        "call_model": call_model,
        "execute_tool": execute_tool,
        "force_finish": force_finish,
    }


def make_config(thread_id, role="write-with-approve"):
    return {
        "configurable": {
            "thread_id": thread_id,
            "user_role": role,
        }
    }


def pauses(snapshot):
    return [
        item
        for task in snapshot.tasks
        for item in task.interrupts
    ]


async def choose(graph, config, bug_id="900002"):
    snapshot = await graph.aget_state(config)
    current = pauses(snapshot)[0]
    await graph.ainvoke(
        Command(resume={current.id: {"bug_id": bug_id}}),
        config,
    )


async def main():
    sender = AsyncMock(return_value={"ok": True})

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        await saver.setup()
        graph = build_agent(saver, nodes=demo_nodes(), sender=sender)

        for name, decision in (("reject", False), ("approve", True)):
            config = make_config(f"{getpass.getuser()}-workaround-{name}")

            await graph.ainvoke(copy.deepcopy(DEMO_INPUT), config)
            selection = await graph.aget_state(config)

            print(f"\nSCENARIO: {name}")
            print(
                "SELECT __interrupt__:",
                json.dumps(pauses(selection)[0].value, ensure_ascii=False),
            )
            assert selection.next == (CHOOSE_NODE,)

            await choose(graph, config)
            confirmation = await graph.aget_state(config)

            print(
                "APPROVE __interrupt__:",
                json.dumps(pauses(confirmation)[0].value, ensure_ascii=False),
            )
            assert confirmation.next == (APPROVE_NODE,)
            sender.assert_not_called()

            history = [
                item async for item in graph.aget_state_history(config)
            ]
            print("checkpoint_id | next")
            for item in history[:5]:
                print(
                    item.config["configurable"]["checkpoint_id"],
                    "|",
                    item.next,
                )

            if name == "reject":
                # Повтор подготовки из checkpoint, где выбор уже сохранён,
                # но окончательное подтверждение ещё не получено.
                before_prepare = next(
                    item for item in history
                    if item.next == ("prepare_send",)
                )
                replay_config = {
                    **config,
                    "configurable": {
                        **config["configurable"],
                        **before_prepare.config["configurable"],
                        "user_role": config["configurable"]["user_role"],
                    },
                }
                await graph.ainvoke(None, replay_config)
                replayed = await graph.aget_state(config)
                print("REPLAY:", replayed.next)
                assert replayed.next == (APPROVE_NODE,)
                assert pauses(replayed)

            result = await graph.ainvoke(
                Command(resume=decision),
                config,
            )
            print(
                "RESULT:",
                result["sent"],
                result["delivery_status"],
                result["selected_bug_id"],
            )
            assert result["sent"] is decision

            old_id = confirmation.config["configurable"]["checkpoint_id"]
            old = await graph.aget_state(
                {
                    "configurable": {
                        "thread_id": config["configurable"]["thread_id"],
                        "checkpoint_id": old_id,
                    }
                }
            )
            print(
                "PAST:",
                "draft =", old.values["draft"],
                "sent =", old.values["sent"],
                "next =", old.next,
            )
            assert old.values["sent"] is False
            assert old.next == (APPROVE_NODE,)

        sender.assert_awaited_once_with(
            {"chat_id": "123456789", "text": SOLUTION_TWO}
        )
        print("\nExternal API calls: 0")
        print("Mock received only the original solution of bug 900002.")


if __name__ == "__main__":
    asyncio.run(main())