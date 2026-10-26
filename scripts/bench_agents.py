"""Пять задач × три реализации × три повтора."""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from scripts.agent_graph_report import make_report
from scripts.compare_agents import terminate_tree


ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = ROOT / "tests/eval/agent_react_tasks.json"
IMPLEMENTATIONS = ("b62", "custom", "prebuilt")
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
POLICY_STOPS = {
    "max_iterations",
    "revision_limit",
    "final_revision_iteration_limit",
    "search_not_performed",
    "empty_answer",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def result_path(folder, task_id, implementation, repeat):
    return folder / f"result-{task_id}-{implementation}-{repeat}.json"


def signature():
    from app.core.config import get_settings

    settings = get_settings()
    paths = [
        *ROOT.glob("app/**/*.py"),
        *ROOT.glob("app/prompts/*.txt"),
        TASKS_PATH,
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "scripts/bench_agents.py",
        ROOT / "scripts/agent_graph_report.py",
    ]
    return {
        "sources": {
            str(path.relative_to(ROOT)).replace("\\", "/"):
                hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))
            if path.exists()
        },
        "versions": {
            name: version(name)
            for name in (
                "openai",
                "langgraph",
                "langchain",
                "langchain-core",
                "langchain-openai",
            )
        },
        "retrieval": {
            "collection_fingerprint": hashlib.sha256(
                settings.ingest_collection.encode("utf-8")
            ).hexdigest(),
            "top_k": settings.retrieval_top_k,
            "score_threshold": settings.rag_score_threshold,
            "request_timeout": settings.llm.request_timeout,
        },
    }


def summarize_calls(calls):
    totals = {}
    for field in TOKEN_FIELDS:
        values = [call.get(field) for call in calls]
        totals[field] = (
            sum(values)
            if values and all(type(v) is int and v >= 0 for v in values)
            else None
        )

    return {
        **totals,
        "usage_complete": all(v is not None for v in totals.values()),
        "total_steps": sum(call.get("phase") == "actor" for call in calls),
    }


async def run_graph(task, implementation, thread_id):
    from app.services.agent_graph import (
        custom_graph,
        make_input,
        prebuilt_graph,
    )

    graph = custom_graph if implementation == "custom" else prebuilt_graph
    state = await graph.ainvoke(
        make_input(task),
        config={
            "recursion_limit": 100,
            "configurable": {"thread_id": thread_id},
        },
    )
    return {
        "answer": state["messages"][-1].content,
        "error": state.get("stop_reason"),
        "llm_calls": state["llm_calls"],
        "tool_results": state["tool_results"],
        "revisions": state["revisions"],
    }


def child_run(args):
    task = next(
        item for item in read_json(args.folder / "tasks.json")
        if item["id"] == args.task
    )
    identity = {
        "task_id": args.task,
        "implementation": args.child,
        "repeat": args.repeat,
    }

    try:
        from app.services.agent_react import _safe_log

        if args.child == "b62":
            from app.services.agent_react import run_agent

            original = run_agent(
                task["task"],
                max_iterations=10,
                timeout_per_iteration_sec=15,
            )
            calls = [
                call
                for step in original["trace"]
                for call in step["llm_calls"]
            ]
            tools = [
                {
                    "name": step["tool_name"],
                    "args": step["tool_args"],
                    "result": json.loads(step["tool_result"]),
                }
                for step in original["trace"]
                if step.get("tool_result")
            ]
            data = {
                "answer": original["answer"],
                "error": original.get("error"),
                "startup": original.get("startup"),
                "llm_calls": calls,
                "tool_results": tools,
                "revisions": original["revisions"],
            }
        else:
            data = asyncio.run(
                run_graph(
                    task["task"],
                    args.child,
                    f"bench-{args.task}-{args.child}-{args.repeat}",
                )
            )

        result = {
            **identity,
            **data,
            **summarize_calls(data["llm_calls"]),
            "tool_error": any(
                item["result"].get("ok") is False
                for item in data["tool_results"]
            ),
        }
        result = _safe_log(result)

    except Exception as exc:
        result = {
            **identity,
            "error": type(exc).__name__,
            "answer": "",
            "tool_error": False,
            "usage_complete": False,
            "total_steps": None,
            **{field: None for field in TOKEN_FIELDS},
        }

    write_json(
        result_path(args.folder, args.task, args.child, args.repeat),
        result,
    )


def launch(folder, task_id, implementation, repeat):
    destination = result_path(folder, task_id, implementation, repeat)
    command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.bench_agents",
        "--child",
        implementation,
        "--folder",
        str(folder),
        "--task",
        task_id,
        "--repeat",
        str(repeat),
    ]

    # Тексты документов и учебной отправки не пишем в сырой stdout-лог.
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=os.name != "nt",
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        ),
    )
    failure = None

    try:
        if process.wait(timeout=600) != 0:
            failure = "runner_error"
    except subprocess.TimeoutExpired:
        terminate_tree(process)
        failure = "runner_timeout"
    except KeyboardInterrupt:
        terminate_tree(process)
        raise

    latency_ms = (time.perf_counter() - started) * 1000

    if failure or not destination.exists():
        result = {
            "task_id": task_id,
            "implementation": implementation,
            "repeat": repeat,
            "error": failure or "missing_result",
            "answer": "",
            "tool_error": False,
            "usage_complete": False,
            "total_steps": None,
            **{field: None for field in TOKEN_FIELDS},
        }
    else:
        result = read_json(destination)

    result["latency_ms"] = round(latency_ms, 2)
    write_json(destination, result)
    return result


def needs_attention(result):
    error = result.get("error")
    return (
        bool(error and error not in POLICY_STOPS)
        or result.get("tool_error", False)
        or not result.get("usage_complete", False)
    )


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--resume", type=Path)
    mode.add_argument("--child", choices=IMPLEMENTATIONS)
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--repeat", type=int)
    args = parser.parse_args()

    if args.child:
        if args.folder is None or args.task is None or args.repeat is None:
            parser.error("--child требует --folder, --task и --repeat")
        child_run(args)
        return

    tasks = read_json(TASKS_PATH)
    if [item["id"] for item in tasks] != ["S1", "S2", "M1", "M2", "P1"]:
        raise SystemExit("Ожидаются исходные задачи S1, S2, M1, M2, P1")

    if args.list:
        for task in tasks:
            print(f"{task['id']}: {task['title']}")
        print("План: 5 задач x 3 реализации x 3 повтора = 45 запусков.")
        print("Внешних запросов не было.")
        return

    for filename in ("agent-graph-custom.mmd", "agent-graph-prebuilt.mmd"):
        if not (ROOT / "docs" / filename).exists():
            raise SystemExit("Сначала выполните scripts.visualize_graph")

    current = signature()

    if args.resume:
        folder = args.resume.resolve()
        metadata = read_json(folder / "metadata.json")
        if metadata["signature"] != current:
            raise SystemExit(
                "Исходники или настройки изменились. "
                "Не смешивайте результаты: начните новый --run."
            )
        if read_json(folder / "tasks.json") != tasks:
            raise SystemExit("Список задач изменился")
    else:
        from app.services.agent_react import MAIN_MODEL, PREMIUM_MODEL

        now = datetime.now(timezone.utc)
        folder = ROOT / "docs/agent-graph-runs" / now.strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        folder.mkdir(parents=True)
        metadata = {
            "started_utc": now.isoformat(),
            "model": MAIN_MODEL,
            "premium_model": PREMIUM_MODEL,
            "versions": current["versions"],
            "signature": current,
        }
        write_json(folder / "metadata.json", metadata)
        write_json(folder / "tasks.json", tasks)

        previous = ROOT / "docs/agent-graph-report.md"
        if previous.exists():
            (folder / "previous-report.md").write_bytes(previous.read_bytes())

    print(f"Результаты: {folder}", flush=True)
    # Начинаем с провокационной задачи: сразу проверяем отказ без tools.
    ordered_tasks = sorted(tasks, key=lambda item: item["id"] != "P1")

    try:
        for repeat in range(1, 4):
            shift = repeat - 1
            order = IMPLEMENTATIONS[shift:] + IMPLEMENTATIONS[:shift]

            for task in ordered_tasks:
                for implementation in order:
                    path = result_path(
                        folder, task["id"], implementation, repeat
                    )
                    if path.exists() and not needs_attention(read_json(path)):
                        continue

                    print(
                        f"{task['id']} / {implementation} / повтор {repeat}",
                        flush=True,
                    )
                    result = launch(
                        folder, task["id"], implementation, repeat
                    )
                    make_report(folder)
                    print(
                        f"  {result['latency_ms']:.0f} мс; "
                        f"шагов: {result['total_steps']}; "
                        f"остановка: {result.get('error') or 'нет'}",
                        flush=True,
                    )

                    if needs_attention(result):
                        print(
                            "Остановка бенчмарка: ошибка или неполный учёт "
                            "токенов. Результаты сохранены.",
                            flush=True,
                        )
                        return
    finally:
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["sources_unchanged"] = signature() == metadata["signature"]
        write_json(folder / "metadata.json", metadata)
        make_report(folder)
        print("Отчёт: docs/agent-graph-report.md", flush=True)


if __name__ == "__main__":
    main()