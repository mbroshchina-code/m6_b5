"""SSE, выбор решения и подтверждение отправки."""

import json
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from app.admin.auth import require_admin


router = APIRouter(
    prefix="/agent",
    tags=["agent"],
    dependencies=[Depends(require_admin)],
)

Identifier = Annotated[
    str,
    Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_.:-]+$"),
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UserMessage(StrictModel):
    role: Literal["user"]
    content: str = Field(min_length=1, max_length=12000)


class AgentInput(StrictModel):
    request_id: Identifier
    messages: list[UserMessage] = Field(min_length=1, max_length=1)
    send_to: str | None = Field(default=None, pattern=r"^-?[1-9]\d*$")


class Selection(StrictModel):
    bug_id: str = Field(pattern=r"^[1-9]\d*$")


class StreamRequest(StrictModel):
    thread_id: Identifier
    input: AgentInput | None = None
    resume: StrictBool | Selection | None = None
    interrupt_id: str | None = None

    @model_validator(mode="after")
    def validate_operation(self):
        if (self.input is None) == (self.resume is None):
            raise ValueError("Передайте либо input, либо resume")
        if self.resume is not None and not self.interrupt_id:
            raise ValueError("Для resume нужен interrupt_id")
        if self.input is not None and self.interrupt_id is not None:
            raise ValueError("interrupt_id используется только с resume")
        return self


def make_config(request, thread_id):
    return {
        "configurable": {
            "thread_id": thread_id,
            "user_role": request.app.state.agent_settings.agent_user_role,
        },
        "recursion_limit": 64,
    }


def pending_interrupts(snapshot):
    return [
        {"id": item.id, "value": item.value}
        for task in snapshot.tasks
        for item in task.interrupts
    ]


def validate_resume(pending, value):
    kind = pending["value"]["type"]

    if kind == "select_workaround":
        if value is False:
            return
        allowed = {
            item["bug_id"] for item in pending["value"]["options"]
        }
        if (
            isinstance(value, dict)
            and set(value) == {"bug_id"}
            and isinstance(value["bug_id"], str)
            and value["bug_id"] in allowed
        ):
            return

        raise HTTPException(422, "Выберите номер из предложенного списка")

    if kind == "approve_send_workaround" and type(value) is bool:
        return

    raise HTTPException(422, "Для этой паузы нужно подтверждение true/false")


def public_state(snapshot):
    values = snapshot.values
    draft = values.get("draft") or {}
    sent = values.get("sent", False)

    return {
        "exists": bool(values),
        "next": list(snapshot.next),
        "interrupts": pending_interrupts(snapshot),
        "request_id": values.get("request_id"),
        "operator_answer": values.get("operator_answer", ""),
        "selected_bug_id": values.get("selected_bug_id"),
        "sent": sent,
        "client_text": draft.get("text") if sent else None,
        "delivery_status": values.get("delivery_status", "not_requested"),
    }


def sse(payload):
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


@router.get("/state")
async def get_state(
    request: Request,
    thread_id: str = Query(
        min_length=1,
        max_length=120,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    ),
):
    graph = request.app.state.agent
    snapshot = await graph.aget_state(make_config(request, thread_id))
    return public_state(snapshot)


@router.post("/stream")
async def stream_agent(body: StreamRequest, request: Request):
    graph = request.app.state.agent
    lock = request.app.state.agent_lock

    if lock.locked():
        raise HTTPException(409, "Агент занят. Дождитесь завершения запроса")

    await lock.acquire()
    config = make_config(request, body.thread_id)

    try:
        snapshot = await graph.aget_state(config)
        pending = pending_interrupts(snapshot)

        if body.input is not None:
            if snapshot.next:
                raise HTTPException(409, "В сессии есть незавершённый запуск")
            if body.input.request_id in snapshot.values.get("seen_requests", []):
                raise HTTPException(409, "Это request_id уже использовалось")

            graph_input = body.input.model_dump()

        else:
            if len(pending) != 1:
                raise HTTPException(409, "Нет единственного ожидающего решения")
            if body.interrupt_id != pending[0]["id"]:
                raise HTTPException(409, "Ответ относится к другой паузе")

            value = (
                body.resume.model_dump()
                if isinstance(body.resume, Selection)
                else body.resume
            )
            validate_resume(pending[0], value)

            # Решение адресуется именно текущему interrupt.
            graph_input = Command(
                resume={pending[0]["id"]: value}
            )

    except BaseException:
        lock.release()
        raise

    async def events():
        try:
            yield sse({"type": "started"})

            async for stream_type, payload in graph.astream(
                graph_input,
                config=config,
                stream_mode=["updates", "messages"],
                durability="sync",
            ):
                if stream_type == "updates":
                    for node, update in payload.items():
                        if node == "__interrupt__":
                            for item in update:
                                yield sse(
                                    {
                                        "type": "interrupt",
                                        "id": item.id,
                                        "value": item.value,
                                    }
                                )
                        else:
                            yield sse({"type": "update", "node": node})

                elif stream_type == "messages":
                    message, metadata = payload
                    if (
                        metadata.get("langgraph_node") == "call_model"
                        and isinstance(message.content, str)
                        and message.content
                    ):
                        yield sse(
                            {
                                "type": "draft_token",
                                "text": message.content,
                                "final": False,
                            }
                        )

            snapshot = await graph.aget_state(config)
            result = public_state(snapshot)
            yield sse({"type": "result", **result})
            yield sse(
                {
                    "type": (
                        "paused"
                        if result["interrupts"]
                        else "incomplete"
                        if snapshot.next
                        else "done"
                    )
                }
            )

        except Exception:
            yield sse(
                {
                    "type": "error",
                    "message": "Обработка прервана. Проверьте /agent/state",
                }
            )
        finally:
            lock.release()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )