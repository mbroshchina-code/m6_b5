"""Один платный контрольный запуск ReAct на задаче S1."""

import json
from datetime import datetime, timezone
from pathlib import Path

from app.services.agent_react import _safe_log, run_agent


ROOT = Path(__file__).resolve().parents[1]


def main():
    tasks = json.loads(
        (ROOT / "tests/eval/agent_react_tasks.json").read_text(
            encoding="utf-8-sig"
        )
    )
    case = next(item for item in tasks if item["id"] == "S1")

    folder = ROOT / "docs/agent-react-checks"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    destination = folder / f"S1-{stamp}.json"

    print("Запускается один платный прогон ReAct / S1.", flush=True)
    result = run_agent(
        case["task"],
        max_iterations=10,
        timeout_per_iteration_sec=15,
    )

    record = {
        "case_id": case["id"],
        "task": case["task"],
        "result": result,
    }
    destination.write_text(
        json.dumps(_safe_log(record), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Ответ:", result["answer"])
    print("Ошибка:", result.get("error", "нет"))
    print("Итераций:", result["steps"])
    print("Ревизий:", result["revisions"])
    print("Токены:", result["usage"])
    print("Учёт токенов полный:", result["usage_complete"])
    print("Результат сохранён:", destination)

    if result.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()