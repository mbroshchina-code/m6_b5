import copy
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.routers.agent import validate_resume
from app.services.agent_persistent import (
    APPROVE_NODE,
    CHOOSE_NODE,
    build_agent,
)
from app.services.agent_persistent_cards import extract_field
from scripts.time_travel_demo import (
    DEMO_INPUT,
    SOLUTION_TWO,
    choose,
    demo_nodes,
    make_config,
    pauses,
)


@pytest.mark.asyncio
async def test_requires_selection_and_then_confirmation():
    sender = AsyncMock(return_value={"ok": True})

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        await saver.setup()
        graph = build_agent(saver, nodes=demo_nodes(), sender=sender)
        config = make_config("test-pauses")

        await graph.ainvoke(copy.deepcopy(DEMO_INPUT), config)
        snapshot = await graph.aget_state(config)

        assert snapshot.next == (CHOOSE_NODE,)
        assert {item["bug_id"] for item in snapshot.values["options"]} == {
            "900001", "900002"
        }
        sender.assert_not_called()

        await choose(graph, config)
        snapshot = await graph.aget_state(config)

        assert snapshot.next == (APPROVE_NODE,)
        assert pauses(snapshot)
        assert snapshot.values["draft"]["text"] == SOLUTION_TWO
        assert snapshot.values["sent"] is False
        sender.assert_not_called()


@pytest.mark.asyncio
async def test_approval_sends_only_selected_source_solution():
    sender = AsyncMock(return_value={"ok": True})

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        await saver.setup()
        graph = build_agent(saver, nodes=demo_nodes(), sender=sender)
        config = make_config("test-approve")

        await graph.ainvoke(copy.deepcopy(DEMO_INPUT), config)
        await choose(graph, config)
        result = await graph.ainvoke(Command(resume=True), config)

        assert result["sent"] is True
        assert result["selected_bug_id"] == "900002"
        sender.assert_awaited_once_with(
            {"chat_id": "123456789", "text": SOLUTION_TWO}
        )


@pytest.mark.asyncio
async def test_rejection_never_calls_sender():
    sender = AsyncMock(return_value={"ok": True})

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        await saver.setup()
        graph = build_agent(saver, nodes=demo_nodes(), sender=sender)
        config = make_config("test-reject")

        await graph.ainvoke(copy.deepcopy(DEMO_INPUT), config)
        await choose(graph, config)
        result = await graph.ainvoke(Command(resume=False), config)

        assert result["sent"] is False
        assert result["delivery_status"] == "rejected"
        sender.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_selection_never_calls_sender():
    sender = AsyncMock(return_value={"ok": True})

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        await saver.setup()
        graph = build_agent(saver, nodes=demo_nodes(), sender=sender)
        config = make_config("test-cancel")

        await graph.ainvoke(copy.deepcopy(DEMO_INPUT), config)
        result = await graph.ainvoke(Command(resume=False), config)

        assert result["delivery_status"] == "cancelled"
        sender.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_choice_is_rejected_before_resume():
    sender = AsyncMock(return_value={"ok": True})

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        await saver.setup()
        graph = build_agent(saver, nodes=demo_nodes(), sender=sender)
        config = make_config("test-invalid")

        await graph.ainvoke(copy.deepcopy(DEMO_INPUT), config)
        before = await graph.aget_state(config)
        current = pauses(before)[0]
        pending = {"id": current.id, "value": current.value}

        # 900003 существует, но относится к менее релевантным.
        for value in (
            {"bug_id": "900003"},
            {"bug_id": "999999"},
            {"bug_id": "900002", "text": "Подменённый текст"},
            True,
        ):
            with pytest.raises(HTTPException) as error:
                validate_resume(pending, value)
            assert error.value.status_code == 422

        after = await graph.aget_state(config)
        assert pauses(after)[0].id == current.id
        assert after.next == (CHOOSE_NODE,)
        sender.assert_not_called()


@pytest.mark.asyncio
async def test_read_only_cannot_send():
    sender = AsyncMock(return_value={"ok": True})

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        await saver.setup()
        graph = build_agent(saver, nodes=demo_nodes(), sender=sender)
        config = make_config("test-read-only", role="read-only")

        result = await graph.ainvoke(copy.deepcopy(DEMO_INPUT), config)

        assert result["delivery_status"] == "forbidden"
        assert result["sent"] is False
        sender.assert_not_called()


@pytest.mark.asyncio
async def test_no_matching_bugs_does_not_offer_selection():
    sender = AsyncMock(return_value={"ok": True})

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        await saver.setup()
        graph = build_agent(
            saver,
            nodes=demo_nodes(answer="Подходящих багов не найдено", cards=[]),
            sender=sender,
        )
        config = make_config("test-no-match")

        result = await graph.ainvoke(copy.deepcopy(DEMO_INPUT), config)
        snapshot = await graph.aget_state(config)

        assert result["operator_answer"] == "Подходящих багов не найдено"
        assert result["options"] == []
        assert not snapshot.next
        sender.assert_not_called()


@pytest.mark.asyncio
async def test_reopen_sqlite_preserves_selected_solution(tmp_path):
    database = str(tmp_path / "checkpoints.sqlite3")
    config = make_config("test-restart")
    sender = AsyncMock(return_value={"ok": True})

    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        await saver.setup()
        graph = build_agent(saver, nodes=demo_nodes(), sender=sender)

        await graph.ainvoke(copy.deepcopy(DEMO_INPUT), config)
        await choose(graph, config)
        before = await graph.aget_state(config)

    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        await saver.setup()
        graph = build_agent(saver, nodes=demo_nodes(), sender=sender)
        after = await graph.aget_state(config)

        assert after.next == (APPROVE_NODE,)
        assert after.values["selected_bug_id"] == "900002"
        assert after.values["draft"] == before.values["draft"]

        result = await graph.ainvoke(Command(resume=True), config)

        assert result["sent"] is True
        sender.assert_awaited_once_with(
            {"chat_id": "123456789", "text": SOLUTION_TWO}
        )


@pytest.mark.parametrize("prefix", ["## ", ""])
def test_extracts_multiline_solution_without_next_field(prefix):
    text = (
        f"{prefix}Временное решение\n"
        "Первый шаг.\n\nВторой шаг.\n"
        f"{prefix}Статус\nОткрыт\n"
    )
    assert extract_field(text, "временное решение") == (
        "Первый шаг.\n\nВторой шаг."
    )
    
def test_extract_solution_with_repeated_card_title():
    prefix = "Баг №101: Тестовый баг"
    text = (
        f"{prefix} Номер бага\n"
        "101\n\n"
        f"{prefix} Название\n"
        "Тестовый баг\n\n"
        f"{prefix} Временное решение\n"
        "Первая строка решения.\n\n"
        "Вторая строка решения.\n\n"
        f"{prefix} Примечания\n"
        "Это не часть решения.\n"
    )

    assert extract_field(text, "номер бага") == "101"
    assert extract_field(text, "название") == "Тестовый баг"
    assert extract_field(text, "временное решение") == (
        "Первая строка решения.\n\nВторая строка решения."
    )