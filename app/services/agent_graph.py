"""Custom StateGraph и prebuilt-агент с одинаковыми tools и лимитом."""

import json
import operator
from typing import Annotated, Literal, TypedDict
from app.services import agent_graph_reflection as reflection

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState as PrebuiltBaseState,
    after_model,
    before_model,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.services.agent_graph_model import make_graph_model
from app.services.agent_graph_tools import TOOLS
from app.services.agent_graph_prompts import ACTOR_RULES


MAX_ITERATIONS = 6
SYSTEM_PROMPT = ACTOR_RULES

LIMIT_ANSWER = (
    "Достигнут лимит шагов. Обработка обращения не завершена. "
    "Требуется проверка оператором."
)
EMPTY_ANSWER = (
    "Модель не сформировала итоговый ответ. "
    "Требуется проверка оператором."
)

model = make_graph_model()
bound_model = model.bind_tools(
    TOOLS,
    strict=True,
    parallel_tool_calls=False,
)
TOOLS_BY_NAME = {item.name: item for item in TOOLS}


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration_count: int
    tool_results: Annotated[list[dict], operator.add]
    revisions: int
    premium_next: bool
    stop_reason: str | None
    llm_calls: Annotated[list[dict], operator.add]


class PrebuiltState(PrebuiltBaseState):
    iteration_count: int
    tool_results: Annotated[list[dict], operator.add]
    revisions: int
    premium_next: bool
    stop_reason: str | None
    llm_calls: Annotated[list[dict], operator.add]


def make_input(
    task: str,
    history: list[dict] | None = None,
) -> AgentState:
    """Подготовить сериализуемое состояние одного запуска."""
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Нужна непустая задача")

    messages = []

    for item in history or []:
        role = item.get("role")
        content = item.get("content")

        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise ValueError("История должна содержать текст user/assistant")

        message_class = HumanMessage if role == "user" else AIMessage
        messages.append(message_class(content=content))

    messages.append(HumanMessage(content=task))

    return {
        "messages": messages,
        "iteration_count": 0,
        "tool_results": [],
        "revisions": 0,
        "premium_next": False,
        "stop_reason": None,
        "llm_calls": [],
    }


def _dump(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False)


def _finish_update(messages: list[AnyMessage]) -> dict:
    """Завершить незакрытые tool calls и обеспечить текстовый ответ."""
    last = messages[-1]
    calls = getattr(last, "tool_calls", None) or []

    if calls:
        cancelled = [
            ToolMessage(
                content=_dump({
                    "ok": False,
                    "error": "Вызов не выполнен: достигнут лимит шагов",
                }),
                tool_call_id=call["id"],
                name=call["name"],
                status="error",
            )
            for call in calls
        ]
        return {
            "messages": [
                *cancelled,
                AIMessage(content=LIMIT_ANSWER),
            ],
            "stop_reason": "max_iterations",
        }

    if not isinstance(last, AIMessage) or not last.content:
        return {
            "messages": [AIMessage(content=EMPTY_ANSWER)],
            "stop_reason": "empty_answer",
        }

    # Готовый ответ уже находится в state: повторно добавлять его не нужно.
    return {}


async def call_model(state: AgentState) -> dict:
    if state.get("stop_reason"):
        return {
            "messages": [AIMessage(content=reflection.STOP_ANSWER)],
            "premium_next": False,
        }

    selected = bound_model
    if state.get("premium_next"):
        selected = reflection.premium_model.bind_tools(
            TOOLS,
            strict=True,
            parallel_tool_calls=False,
        )

    response = await selected.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"],
    ])

    return {
        "messages": [response],
        "iteration_count": state["iteration_count"] + 1,
        "premium_next": False,
        "llm_calls": [reflection.usage_record(response, "actor")],
    }


async def execute_tool(state: AgentState) -> dict:
    calls = getattr(state["messages"][-1], "tool_calls", None) or []
    messages = []
    results = []

    for call in calls:
        name = call["name"]
        selected_tool = TOOLS_BY_NAME.get(name)

        if selected_tool is None:
            outcome = {
                "ok": False,
                "error": "Неизвестный инструмент",
            }
        else:
            try:
                value = await selected_tool.ainvoke(call["args"])
                outcome = {"ok": True, "value": str(value)}
            except Exception as exc:
                # Текст исключения может содержать адреса и секреты.
                outcome = {"ok": False, "error": type(exc).__name__}

        content = _dump(outcome)

        messages.append(
            ToolMessage(
                content=content,
                tool_call_id=call["id"],
                name=name,
                status="success" if outcome["ok"] else "error",
            )
        )
        results.append({
            "tool_call_id": call["id"],
            "name": name,
            "args": call["args"],
            "result": outcome,
        })

    update = {
        "messages": messages,
        "tool_results": results,
    }
    working = reflection.merge_updates(state, update)
    assessment = await reflection.review(working, "observation")

    return reflection.merge_updates(update, assessment)


async def force_finish(state: AgentState) -> dict:
    if state.get("stop_reason"):
        return {}

    basic_finish = _finish_update(state["messages"])
    if basic_finish:
        return basic_finish

    invalid = reflection.final_answer_guard(state["messages"])
    if invalid:
        return invalid

    delivery_error = reflection.delivery_confirmation_guard(state)
    if delivery_error:
        return delivery_error

    # Без выполненных tools отдельный критик не нужен — как в Б6.2.
    if not state.get("tool_results"):
        return {}

    return await reflection.finish_with_review(
        state,
        MAX_ITERATIONS,
    )


def route_after_model(
    state: AgentState,
) -> Literal["execute_tool", "force_finish"]:
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "force_finish"

    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "execute_tool"

    return "force_finish"


def build_custom_graph(*, checkpointer=None):
    builder = StateGraph(AgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_tool", execute_tool)
    builder.add_node("force_finish", force_finish)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        route_after_model,
        {
            "execute_tool": "execute_tool",
            "force_finish": "force_finish",
        },
    )
    builder.add_edge("execute_tool", "call_model")
    builder.add_edge("force_finish", END)

    return builder.compile(checkpointer=checkpointer)


@wrap_tool_call
async def safe_prebuilt_tool(request, handler):
    """Вернуть ошибку инструмента модели, сохранив tool_call_id."""
    call = request.tool_call

    try:
        result = await handler(request)
        if not isinstance(result, ToolMessage):
            raise TypeError("Ожидался ToolMessage")

        if result.status == "error":
            outcome = {
                "ok": False,
                "error": "Ошибка выполнения или проверки аргументов",
            }
        else:
            outcome = {"ok": True, "value": str(result.content)}
    except Exception as exc:
        outcome = {"ok": False, "error": type(exc).__name__}

    return ToolMessage(
        content=_dump(outcome),
        tool_call_id=call["id"],
        name=call["name"],
        status="success" if outcome["ok"] else "error",
    )


def _new_tool_results(state: PrebuiltState) -> list[dict]:
    recorded_ids = {
        item["tool_call_id"]
        for item in state.get("tool_results", [])
    }
    calls_by_id = {
        call["id"]: call
        for message in state["messages"]
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    results = []

    for message in state["messages"]:
        if not isinstance(message, ToolMessage):
            continue
        if message.tool_call_id in recorded_ids:
            continue

        call = calls_by_id.get(message.tool_call_id, {})
        results.append({
            "tool_call_id": message.tool_call_id,
            "name": message.name or call.get("name", ""),
            "args": call.get("args", {}),
            "result": json.loads(message.content),
        })
        recorded_ids.add(message.tool_call_id)

    return results


@before_model(
    state_schema=PrebuiltState,
    can_jump_to=["end"],
)
async def review_prebuilt_observation(state: PrebuiltState, runtime):
    results = _new_tool_results(state)
    if not results:
        return None

    update = {"tool_results": results}
    working = reflection.merge_updates(state, update)
    assessment = await reflection.review(working, "observation")
    update = reflection.merge_updates(update, assessment)

    if update.get("stop_reason"):
        update = reflection.merge_updates(
            update,
            {
                "messages": [AIMessage(content=reflection.STOP_ANSWER)],
                "jump_to": "end",
            },
        )

    return update


@wrap_model_call
async def select_prebuilt_model(request, handler):
    selected = (
        reflection.premium_model
        if request.state.get("premium_next")
        else model
    )

    return await handler(
        request.override(
            model=selected,
            model_settings={
                **request.model_settings,
                "parallel_tool_calls": False,
            },
        )
    )


@after_model(
    state_schema=PrebuiltState,
    can_jump_to=["end"],
)
async def track_prebuilt_step(state: PrebuiltState, runtime) -> dict:
    response = state["messages"][-1]

    update = {
        "iteration_count": state.get("iteration_count", 0) + 1,
        "premium_next": False,
        "llm_calls": [reflection.usage_record(response, "actor")],
    }
    working = reflection.merge_updates(state, update)

    if (
        working["iteration_count"] >= MAX_ITERATIONS
        or not getattr(response, "tool_calls", None)
    ):
        finished = await force_finish(working)
        update = reflection.merge_updates(update, finished)
        update["jump_to"] = "end"

    return update


def build_prebuilt_graph(*, checkpointer=None):
    return create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        state_schema=PrebuiltState,
        middleware=[
            safe_prebuilt_tool,
            review_prebuilt_observation,
            select_prebuilt_model,
            track_prebuilt_step,
        ],
        checkpointer=checkpointer,
    )


custom_graph = build_custom_graph()
prebuilt_graph = build_prebuilt_graph()