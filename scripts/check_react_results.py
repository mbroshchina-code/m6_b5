"""Проверка сохранённых результатов. Внешних запросов нет."""

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = ("S1", "S2", "M1", "M2", "P1")
FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Укажите папку сравнительного запуска")

    folder = Path(sys.argv[1])
    metadata = read_json(folder / "metadata.json")

    # Файлы эксперимента не менялись во время и после запуска.
    for name, expected_hash in metadata["files_before"].items():
        assert metadata["files_after"][name] == expected_hash, (
            f"Файл изменился во время эксперимента: {name}"
        )
        actual_hash = hashlib.sha256(
            (ROOT / name).read_bytes()
        ).hexdigest()
        assert actual_hash == expected_hash, (
            f"Файл изменился после эксперимента: {name}"
        )

    totals = {"naive": 0, "react": 0}

    for case in CASES:
        for agent in totals:
            result = read_json(folder / f"{case}-{agent}.json")
            assert not result.get("error"), (case, agent, result.get("error"))
            assert result.get("answer"), (case, agent, "Пустой ответ")

            trace = result["trace"]
            assert result["steps"] == len(trace), (case, agent)

            if agent == "naive":
                for row in trace:
                    for key in ("llm_input_tokens", "llm_output_tokens"):
                        assert type(row.get(key)) is int, (case, key)
                        assert row[key] >= 0, (case, key)

                totals[agent] += sum(
                    row["llm_input_tokens"] + row["llm_output_tokens"]
                    for row in trace
                )
                continue

            assert result["usage_complete"] is True, case
            assert 0 <= result["revisions"] <= 2, case
            assert result["steps"] <= 10, case
            calls = [
                call
                for row in trace
                for call in row["llm_calls"]
            ]

            for call in calls:
                assert call["usage_complete"] is True, case
                for field in FIELDS:
                    assert type(call[field]) is int, (case, field)
                    assert call[field] >= 0, (case, field)
                assert call["total_tokens"] == (
                    call["prompt_tokens"] + call["completion_tokens"]
                ), case

            for field in FIELDS:
                assert result["usage"][field] == sum(
                    call[field] for call in calls
                ), (case, field)

            totals[agent] += result["usage"]["total_tokens"]

            if case == "P1":
                assert all(
                    row.get("tool_name") is None for row in trace
                ), "В P1 обнаружен вызов инструмента"

    assert totals == {"naive": 20353, "react": 49983}, totals
    print("OK: 10 результатов; суммы токенов совпали; файлы сохранены.")
    print("Naive: 20353; ReAct: 49983.")
    print("Внешних запросов не было.")


if __name__ == "__main__":
    main()