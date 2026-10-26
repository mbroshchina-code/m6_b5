"""Сравнение baseline и ReAct на пяти одинаковых задачах."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = ROOT / "tests/eval/agent_react_tasks.json"
REPORT_PATH = ROOT / "docs/agent-react-report.md"

BASELINE_FILES = [
    "app/services/agent_naive.py",
    "app/services/agent_naive_prompt.py",
    "app/services/agent_naive_tools.py",
]

SOURCE_FILES = BASELINE_FILES + [
    "app/services/agent_react.py",
    "app/services/agent_react_tools.py",
    "app/services/agent_react_runtime.py",
    "app/services/rag.py",
    "app/services/embeddings.py",
    "data/training_bugs/bug_112.md",
    "data/training_bugs/bug_110.md",
    "tests/eval/agent_react_tasks.json",
    "scripts/compare_agents.py",
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def fingerprints():
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }


def result_path(folder, case_id, agent):
    return folder / f"{case_id}-{agent}.json"


def child_run(folder, case_id, agent):
    cases = read_json(folder / "tasks.json")
    case = next(item for item in cases if item["id"] == case_id)

    try:
        # Эталонные сведения из expected модели не передаются.
        if agent == "naive":
            from app.services.agent_naive import run_agent

            result = run_agent(case["task"])
        else:
            from app.services.agent_react import run_agent

            result = run_agent(
                case["task"],
                max_iterations=10,
                timeout_per_iteration_sec=15,
            )
    except Exception as exc:
        result = {
            "answer": f"Ошибка запуска: {type(exc).__name__}",
            "error": type(exc).__name__,
            "steps": None,
            "trace": [],
            "usage_complete": False,
        }

    write_json(result_path(folder, case_id, agent), result)


def terminate_tree(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    else:
        os.killpg(process.pid, signal.SIGKILL)

    process.wait(timeout=10)


def launch(folder, case_id, agent):
    destination = result_path(folder, case_id, agent)
    log_path = folder / f"{case_id}-{agent}.log"
    command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.compare_agents",
        "--child",
        agent,
        "--case-id",
        case_id,
        "--folder",
        str(folder),
    ]
    environment = dict(
        os.environ,
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
    )
    failure = None

    with log_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP
                if os.name == "nt"
                else 0
            ),
        )
        try:
            exit_code = process.wait(timeout=240)
            if exit_code != 0:
                failure = "runner_error"
        except subprocess.TimeoutExpired:
            terminate_tree(process)
            failure = "runner_timeout"
        except KeyboardInterrupt:
            terminate_tree(process)
            raise

    if failure or not destination.exists():
        result = {
            "answer": (
                "Запуск остановлен внешним ограничением времени"
                if failure == "runner_timeout"
                else "Ошибка запуска; подробности в локальном логе"
            ),
            "error": failure or "missing_result",
            "steps": None,
            "trace": [],
            "usage_complete": False,
        }
        write_json(destination, result)

    return read_json(destination)


def token_text(result, agent):
    if agent == "react":
        usage = result.get("usage")
        if not usage:
            return "неизвестно"
        total = usage["total_tokens"]
        return (
            str(total)
            if result.get("usage_complete")
            else f"≥{total}, учёт неполный"
        )

    trace = result.get("trace") or []
    if not trace:
        return "неизвестно"

    total = sum(
        (row.get("llm_input_tokens") or 0)
        + (row.get("llm_output_tokens") or 0)
        for row in trace
    )
    complete = all(
        (row.get("llm_input_tokens") or 0) > 0
        and isinstance(row.get("llm_output_tokens"), int)
        for row in trace
    )
    return str(total) if complete else f"≥{total}, учёт неполный"


def technical_failure(result):
    expected_stops = {
        "max_steps",
        "max_iterations",
        "revision_limit",
        "search_not_performed",
        "empty_answer",
        "tool_protocol_error",
        "invalid_critic",
        "incomplete_response",
    }
    error = result.get("error")
    if error and error not in expected_stops:
        return True

    for row in result.get("trace") or []:
        observation = row.get("tool_result") or ""
        if "Ошибка инструмента:" in observation:
            return True
        try:
            parsed = json.loads(observation)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict) and parsed.get("ok") is False:
            name = parsed.get("error", "")
            if name.endswith(("Error", "Exception")):
                return True
    return False


def make_report(folder):
    from app.services.agent_react import _safe_log

    cases = read_json(folder / "tasks.json")
    metadata = read_json(folder / "metadata.json")
    records = {}
    for case in cases:
        for agent in ("naive", "react"):
            path = result_path(folder, case["id"], agent)
            if path.exists():
                records[(case["id"], agent)] = read_json(path)

    unchanged = all(
        metadata["files_before"].get(name)
        == metadata["files_after"].get(name)
        for name in BASELINE_FILES
    )

    lines = [
        "# Сравнение naive и ReAct",
        "",
        "**Черновик: корректность ответов требует ручной проверки.**",
        "",
        f"Начало эксперимента UTC: {metadata['started_utc']}.",
        f"Результатов сохранено: {len(records)} из 10.",
        f"Версия OpenAI SDK: {metadata['openai_version']}.",
        f"Коллекция: `{metadata['collection']}`.",
        f"Baseline не изменился во время эксперимента: {'да' if unchanged else 'нет'}.",
        "",
        "Naive: исходные настройки, max_steps=6.",
        "Оба агента используют общий модуль embeddings.py с отложенным "
        "импортом SentenceTransformer для локального провайдера. "
        "Файлы baseline не изменялись.",
        "ReAct: max_iterations=10, общий таймаут шага=15 секунд, "
        "максимум две ревизии.",
        "Модель агента и критика — gpt-5.4-mini; "
        "следующий шаг после REVISE — gpt-5.4.",
        "У ReAct также проверяется финальный ответ после работы с tools.",
        "Каждый запуск выполняется в отдельном процессе. "
        "Порядок для каждой задачи: naive, затем react.",
        "Внешний предел одного запуска — 240 секунд; "
        "это отдельное ограничение измерительного скрипта.",
        "Перед циклом ReAct выполняется подготовка с пределом 60 секунд: "
        "загрузка библиотек, создание клиентов и подключение существующего "
        "индекса с проверками Qdrant. Поиск, генерация LLM и расчёт "
        "эмбеддингов в подготовке не выполняются.",
        "Naive подключает индекс при первом поиске, ReAct — до цикла. "
        "Поэтому сравнение времени должно учитывать весь запуск, "
        "включая подготовку ReAct.",
        "Подготовка измеряется отдельно и входит в полное время запуска. "
        "Поиск и критик входят в общий 15-секундный лимит итерации.",
        "",
        "Итерации включают шаг финального ответа. Вызовы критика "
        "не считаются отдельными итерациями, но входят в токены.",
        "Токены — input + output Chat Completions. "
        "Эмбеддинги и денежная стоимость сюда не входят.",
        "Знак ≥ означает неполный учёт, а не точный расход.",
        "",
        "| Задача | Итераций naive | Итераций react | "
        "Корректность naive | Корректность react | "
        "Токенов naive | Токенов react | Ревизия ReAct |",
        "|---|---:|---:|---|---|---|---|---|",
    ]

    for case in cases:
        naive = records.get((case["id"], "naive"))
        react = records.get((case["id"], "react"))

        def steps(result):
            if result is None or result.get("steps") is None:
                return "—"
            return str(result["steps"])

        def tokens(result, agent):
            return "—" if result is None else token_text(result, agent)

        def correctness(result):
            return "не запускалось" if result is None else "не проверено"

        if react is None:
            revision = "—"
        elif "revisions" not in react:
            revision = "неизвестно"
        else:
            count = react["revisions"]
            revision = f"да ({count})" if count else "нет"

        lines.append(
            f"| {case['id']}: {case['title']} "
            f"| {steps(naive)} | {steps(react)} "
            f"| {correctness(naive)} | {correctness(react)} "
            f"| {tokens(naive, 'naive')} | {tokens(react, 'react')} "
            f"| {revision} |"
        )

    lines += [
        "",
        "## Наблюдения — заполнить после проверки",
        "",
        "1. Где критик обнаружил ошибку и что изменилось: не оценено.",
        "2. Где критик не изменил результат, но добавил расход: не оценено.",
        "3. Повторные действия и достижение лимитов: не оценено.",
        "4. Отказ от провокационной задачи без tools: не оценено.",
        "",
        "## Задачи и фактические результаты",
    ]

    for case in cases:
        lines += [
            "",
            f"### {case['id']}: {case['title']}",
            "",
            f"Уровень: {case['level']}.",
            "",
            f"Задача: {case['task']}",
            "",
            f"Ожидание: {case['expected']}",
        ]
        for agent in ("naive", "react"):
            result = records.get((case["id"], agent))
            lines += ["", f"**{agent}**", ""]
            if result is None:
                lines.append("Не запускалось.")
                continue

            details = {
                "answer": result.get("answer"),
                "startup": result.get("startup"),
                "total_latency_ms": result.get("total_latency_ms"),
                "error": result.get("error"),
                "steps": result.get("steps"),
                "tokens": token_text(result, agent),
                "trace": [
                    {
                        "step": row.get("step"),
                        "tool": row.get("tool_name"),
                        "arguments": row.get("tool_args"),
                        "observation": row.get("tool_result"),
                        "critic": row.get("critic"),
                    }
                    for row in result.get("trace") or []
                ],
            }
            lines += [
                "```json",
                json.dumps(
                    _safe_log(details),
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
            ]

    text = "\n".join(lines) + "\n"
    (folder / "report-draft.md").write_text(text, encoding="utf-8")
    REPORT_PATH.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--child", choices=("naive", "react"))
    parser.add_argument("--case-id")
    parser.add_argument("--folder", type=Path)
    args = parser.parse_args()

    if args.child:
        if not args.folder or not args.case_id:
            parser.error("--child требует --folder и --case-id")
        child_run(args.folder, args.case_id, args.child)
        return

    cases = read_json(TASKS_PATH)
    if [case["id"] for case in cases] != ["S1", "S2", "M1", "M2", "P1"]:
        raise SystemExit("Ожидаются пять задач: S1, S2, M1, M2, P1")

    if args.list:
        for case in cases:
            print(f"{case['id']}: {case['level']} — {case['title']}")
        print("OK: пять задач прочитаны; внешних запросов не было.")
        return

    from app.core.config import get_settings

    settings = get_settings()
    now = datetime.now(timezone.utc)
    folder = (
        ROOT
        / "docs/agent-react-runs"
        / now.strftime("%Y%m%d_%H%M%S_%f")
    )
    folder.mkdir(parents=True)
    write_json(folder / "tasks.json", cases)

    if REPORT_PATH.exists():
        (folder / "previous-report.md").write_bytes(
            REPORT_PATH.read_bytes()
        )

    metadata = {
        "started_utc": now.isoformat(),
        "openai_version": version("openai"),
        "collection": settings.ingest_collection,
        "rag_score_threshold": settings.rag_score_threshold,
        "retrieval_top_k": settings.retrieval_top_k,
        "files_before": fingerprints(),
    }
    write_json(folder / "metadata.json", metadata)
    print(f"Результаты: {folder}", flush=True)

    try:
        for case in cases:
            for agent in ("naive", "react"):
                print(f"Запуск {case['id']} / {agent}...", flush=True)
                result = launch(folder, case["id"], agent)
                print(
                    f"  Шаги: {result.get('steps')}; "
                    f"токены: {token_text(result, agent)}; "
                    f"ошибка: {result.get('error', 'нет')}",
                    flush=True,
                )
                if technical_failure(result):
                    print(
                        "Остановка после технического сбоя. "
                        "Остальные платные прогоны не запускаются.",
                        flush=True,
                    )
                    return
    finally:
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["files_after"] = fingerprints()
        write_json(folder / "metadata.json", metadata)
        make_report(folder)
        print(f"Черновик отчёта: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()