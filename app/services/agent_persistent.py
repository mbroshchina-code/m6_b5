"""Persistent-граф: поиск → выбор решения → подтверждение → отправка."""

import asyncio
import hashlib
import json
import operator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, TypedDict
from urllib.parse import quote

from langchain_core.messages import AIMessage, AnyMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Overwrite, interrupt
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.services.agent_persistent_cards import make_options


CHOOSE_NODE = "choose_workaround"
APPROVE_NODE = "confirm_and_execute_send"
MAX_ITERATIONS = 6


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    agent_checkpointer: Literal["memory", "sqlite", "postgres"] = "sqlite"
    agent_sqlite_path: Path = Path("var/agent/workarounds.sqlite3")
    agent_user_role: Literal[
        "read-only", "write-with-approve", "full"
    ] = "write-with-approve"

    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "bag_assistant"
    postgres_user: str = "bag_assistant"
    postgres_password: SecretStr = SecretStr("")

    def postgres_uri(self):
        password = self.postgres_password.get_secret_value()
        if not password:
            raise ValueError("Для Postgres нужен POSTGRES_PASSWORD")

        host = self.postgres_host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"

        return (
            f"postgresql://{quote(self.postgres_user, safe='')}:"
            f"{quote(password, safe='')}@{host}:{self.postgres_port}/"
            f"{quote(self.postgres_db, safe='')}"
        )


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration_count: int
    tool_results: Annotated[list[dict], operator.add]
    revisions: int
    premium_next: bool
    stop_reason: str | None
    llm_calls: Annotated[list[dict], operator.add]

    source_cards: Annotated[list[dict], operator.add]
    seen_requests: Annotated[list[str], operator.add]
    request_id: str
    send_to: str | None

    operator_answer: str
    options: list[dict]
    selected_bug_id: str | None
    draft: dict | None
    action_id: str | None
    sent: bool
    delivery_status: str


@tool
async def send_workaround(chat_id: str, text: str) -> dict:
    """Отправить утверждённое временное решение через API бота."""
    from app.services.notifier import notify_user

    await notify_user(chat_id_tg=int(chat_id), text=text)
    return {"ok": True}


def may_send(config):
    return config["configurable"].get("user_role") in {
        "write-with-approve",
        "full",
    }


def build_agent(checkpointer, *, nodes=None, sender=None, rag=None):
    if nodes is None:
        from app.services.agent_persistent_runtime import make_nodes

        nodes = make_nodes(rag)

    if sender is None:
        sender = send_workaround.ainvoke

    async def begin_turn(state):
        return {
            "messages": [
                RemoveMessage(id=message.id)
                for message in state.get("messages", [])
                if getattr(message, "name", None) == "critic_feedback"
                and message.id
            ],
            "seen_requests": [state["request_id"]],
            "iteration_count": 0,
            "tool_results": Overwrite([]),
            "llm_calls": Overwrite([]),
            "source_cards": Overwrite([]),
            "revisions": 0,
            "premium_next": False,
            "stop_reason": None,
            "operator_answer": "",
            "options": [],
            "selected_bug_id": None,
            "draft": None,
            "action_id": None,
            "sent": False,
            "delivery_status": "not_requested",
        }

    async def force_finish(state):
        update = await nodes["force_finish"](state)
        messages = update.get("messages") or state["messages"]
        last = messages[-1]

        answer = (
            last.content
            if isinstance(last, AIMessage)
            and isinstance(last.content, str)
            and not last.tool_calls
            else ""
        )
        return {**update, "operator_answer": answer}

    def after_model(state):
        if state["iteration_count"] >= MAX_ITERATIONS:
            return "force_finish"
        if getattr(state["messages"][-1], "tool_calls", None):
            return "execute_tool"
        return "force_finish"

    async def prepare_options(state):
        if state.get("stop_reason"):
            return {"options": [], "delivery_status": "not_ready"}

        options = make_options(
            state["operator_answer"],
            state.get("source_cards", []),
        )
        return {
            "options": options,
            "delivery_status": (
                "awaiting_selection" if options else "no_available_solution"
            ),
        }

    async def choose_workaround(state, config: RunnableConfig):
        if not may_send(config):
            return {"delivery_status": "forbidden"}

        choice = interrupt(
            {
                "type": "select_workaround",
                "options": [
                    {
                        "bug_id": item["bug_id"],
                        "name": item["name"],
                        "solution": item["solution"],
                    }
                    for item in state["options"]
                ],
            }
        )

        if choice is False:
            return {"delivery_status": "cancelled"}

        if (
            not isinstance(choice, dict)
            or set(choice) != {"bug_id"}
            or choice["bug_id"] not in {
                item["bug_id"] for item in state["options"]
            }
        ):
            return {"delivery_status": "invalid_selection"}

        return {
            "selected_bug_id": choice["bug_id"],
            "delivery_status": "selected",
        }

    async def prepare_send(state, config: RunnableConfig):
        selected = next(
            item
            for item in state["options"]
            if item["bug_id"] == state["selected_bug_id"]
        )

        # Текст берётся только из сохранённой исходной карточки.
        draft = {
            "chat_id": state["send_to"],
            "text": selected["solution"],
        }
        identity = json.dumps(
            {
                "thread_id": config["configurable"]["thread_id"],
                "request_id": state["request_id"],
                "bug_id": selected["bug_id"],
                "source_hash": selected["source_hash"],
                "draft": draft,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

        return {
            "draft": draft,
            "action_id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            "sent": False,
            "delivery_status": "awaiting_approval",
        }

    async def confirm_and_execute_send(state, config: RunnableConfig):
        if not may_send(config):
            return {"sent": False, "delivery_status": "forbidden"}

        decision = interrupt(
            {
                "type": "approve_send_workaround",
                "bug_id": state["selected_bug_id"],
                "action_id": state["action_id"],
                "preview": state["draft"],
            }
        )

        if decision is not True:
            return {"sent": False, "delivery_status": "rejected"}

        # Единственное место внешнего действия.
        try:
            result = await sender(dict(state["draft"]))
        except Exception:
            return {"sent": False, "delivery_status": "unknown"}

        if not isinstance(result, dict) or result.get("ok") is not True:
            return {"sent": False, "delivery_status": "unknown"}

        return {"sent": True, "delivery_status": "sent"}

    builder = StateGraph(AgentState)
    builder.add_node("begin_turn", begin_turn)
    builder.add_node("call_model", nodes["call_model"])
    builder.add_node("execute_tool", nodes["execute_tool"])
    builder.add_node("force_finish", force_finish)
    builder.add_node("prepare_options", prepare_options)
    builder.add_node(CHOOSE_NODE, choose_workaround)
    builder.add_node("prepare_send", prepare_send)
    builder.add_node(APPROVE_NODE, confirm_and_execute_send)

    builder.add_edge(START, "begin_turn")
    builder.add_edge("begin_turn", "call_model")
    builder.add_conditional_edges(
        "call_model",
        after_model,
        {"execute_tool": "execute_tool", "force_finish": "force_finish"},
    )
    builder.add_edge("execute_tool", "call_model")
    builder.add_conditional_edges(
        "force_finish",
        lambda state: "prepare_options" if state.get("send_to") else END,
        {"prepare_options": "prepare_options", END: END},
    )
    builder.add_conditional_edges(
        "prepare_options",
        lambda state: CHOOSE_NODE if state["options"] else END,
        {CHOOSE_NODE: CHOOSE_NODE, END: END},
    )
    builder.add_conditional_edges(
        CHOOSE_NODE,
        lambda state: "prepare_send" if state["selected_bug_id"] else END,
        {"prepare_send": "prepare_send", END: END},
    )
    builder.add_edge("prepare_send", APPROVE_NODE)
    builder.add_edge(APPROVE_NODE, END)

    return builder.compile(checkpointer=checkpointer)


@asynccontextmanager
async def agent_lifespan(app):
    settings = AgentSettings()

    async with AsyncExitStack() as stack:
        if settings.agent_checkpointer == "memory":
            from langgraph.checkpoint.memory import InMemorySaver

            saver = InMemorySaver()

        elif settings.agent_checkpointer == "sqlite":
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            path = settings.agent_sqlite_path
            path.parent.mkdir(parents=True, exist_ok=True)
            saver = await stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(str(path))
            )

        else:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            saver = await stack.enter_async_context(
                AsyncPostgresSaver.from_conn_string(settings.postgres_uri())
            )

        if settings.agent_checkpointer != "memory":
            await saver.setup()

        app.state.agent = build_agent(
            saver,
            rag=getattr(app.state, "rag", None),
        )
        app.state.agent_settings = settings
        app.state.agent_lock = asyncio.Lock()

        try:
            yield
        finally:
            app.state.agent = None