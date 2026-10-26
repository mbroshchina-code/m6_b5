"""Проверка сохранённого эксперимента без внешних запросов."""

import argparse
import json
import re
from pathlib import Path


IMPLEMENTATIONS = ("b62", "custom", "prebuilt")
TASKS = ("S1", "S2", "M1", "M2", "P1")
FALLBACK = "Подходящих багов не найдено"

CARD = re.compile(
    r"\* +\*\*(?P<id>\d+)\*\* — (?P<title>[^\n]+)\n"
    r" {4}\* +\*Статус:\* (?P<status>[^\n]+)\n"
    r" {4}\* +\*Причина:\* (?P<reason>[^\n]+)\n"
    r" {4}\* +\*Временное решение:\* (?P<solution>[^\n]+)"
)


def check_search_answer(text, bug_id, status, strict):
    errors = []

    if strict:
        prefix = "Релевантные баги\n"
        if not text.startswith(prefix):
            return ["Нет установленного заголовка или есть лишнее вступление"]

        match = CARD.fullmatch(text[len(prefix):].strip("\n"))
        if match is None:
            return ["Ответ не соответствует Markdown-шаблону карточки"]

        if match["id"] != bug_id:
            errors.append("Неверный номер бага")
        if match["status"] != status:
            errors.append("Неверный статус")
    else:
        # Б6.2 сохраняет прежний учебный формат.
        if f"**{bug_id}**" not in text or status not in text:
            errors.append("Нет ожидаемого номера или статуса")

    return errors


def check_result(row):
    errors = []
    task = row["task_id"]
    strict = row["implementation"] != "b62"
    answer = row.get("answer", "").strip()
    tools = row.get("tool_results", [])
    names = [item["name"] for item in tools]

    if row.get("error"):
        errors.append(f"Остановка: {row['error']}")
    if not row.get("usage_complete"):
        errors.append("Неполный учёт токенов")
    if any(item["result"].get("ok") is not True for item in tools):
        errors.append("Ошибка инструмента")
    if re.match(r"^(?:OK\s*$|REVISE\s*:)", answer, re.I):
        errors.append("Пользователю выдан служебный ответ")

    if task in {"S1", "S2"}:
        if names != ["search_knowledge_base"]:
            errors.append("Ожидался один поиск")

        bug_id, status = (
            ("112", "Создана") if task == "S1"
            else ("110", "Закрыто")
        )
        errors += check_search_answer(answer, bug_id, status, strict)

    elif task == "M1":
        if names != ["search_knowledge_base", "send_telegram_message"]:
            errors.append("Нарушена цепочка поиск → отправка")

        sends = [
            item for item in tools
            if item["name"] == "send_telegram_message"
        ]
        if len(sends) == 1:
            sent = sends[0]["args"].get("text", "")
            required = (
                "112",
                "Чек аннулирован",
                "Создана",
                "визуаль",
                "не обращать",
            )
            if not all(value in sent for value in required):
                errors.append("В отправке нет ожидаемых сведений о баге 112")

    elif task == "M2":
        if names != ["get_current_time", "send_telegram_message"]:
            errors.append("Нарушена цепочка время → отправка")
        else:
            if tools[0]["args"].get("timezone") != "Europe/Moscow":
                errors.append("Неверный часовой пояс")

            timestamp = tools[0]["result"].get("value", "")
            sent = tools[1]["args"].get("text", "")
            if not timestamp or timestamp not in sent:
                errors.append("Отправлена другая отметка времени")

    elif task == "P1":
        if tools:
            errors.append("Для стороннего запроса вызван инструмент")

        if strict:
            if answer != FALLBACK:
                errors.append("Отказ не соответствует текущему промпту")
        elif "Я помогаю искать известные баги по обращениям клиентов" not in answer:
            errors.append("Нет ожидаемого отказа Б6.2")

    if task in {"M1", "M2"}:
        expected = "Отправка в чат 12345 выполнена в учебном режиме"
        if strict and answer != expected:
            errors.append("Неверный формат подтверждения отправки")
        elif not strict and expected not in answer:
            errors.append("Нет подтверждения учебной отправки")

        for item in tools:
            if item["name"] == "send_telegram_message":
                if item["args"].get("chat_id") != "12345":
                    errors.append("Отправка адресована другому чату")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()

    expected = {
        (task, implementation, repeat)
        for task in TASKS
        for implementation in IMPLEMENTATIONS
        for repeat in range(1, 4)
    }

    rows = [
        json.loads(path.read_text(encoding="utf-8-sig"))
        for path in sorted(args.folder.glob("result-*.json"))
    ]
    actual = {
        (row["task_id"], row["implementation"], row["repeat"])
        for row in rows
    }

    if actual != expected or len(rows) != 45:
        raise SystemExit("Нужны ровно 45 уникальных результатов эксперимента")

    passed = {name: 0 for name in IMPLEMENTATIONS}

    for row in rows:
        errors = check_result(row)
        label = (
            f"{row['task_id']} / {row['implementation']} "
            f"/ повтор {row['repeat']}"
        )

        if errors:
            print("FAIL:", label)
            for error in errors:
                print("  -", error)
        else:
            passed[row["implementation"]] += 1
            print("OK:", label)

    print("\nПроверки формата и действий:")
    for implementation, count in passed.items():
        print(f"{implementation}: {count}/15")

    print("Смысловую достоверность ответов дополнительно проверяем вручную.")


if __name__ == "__main__":
    main()