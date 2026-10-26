"""Извлечение решений из карточек и подготовка вариантов выбора."""

import re
from collections import defaultdict


FIELD_NAMES = {
    "номер бага",
    "название",
    "дата создания",
    "дата создания бага",
    "тема",
    "статус",
    "код статуса",
    "влияние",
    "описание",
    "временное решение",
    "примечание",
    "примечания",
    "источники",
    "ссылки",
}

EMPTY_SOLUTIONS = {
    "",
    "-",
    "—",
    "нет",
    "не указано",
    "не указан",
    "отсутствует",
    "нет временного решения",
}

CARD_LINE = re.compile(
    r"^[*+-]\s+\*\*\[?(\d+)\]?\*\*\s*[—–-]\s*\S"
)


def label(line: str) -> str:
    return line.strip().strip("#*_ :").casefold()


def heading_level(line: str) -> int:
    match = re.match(r"^\s{0,3}(#{1,6})[ \t]+", line)
    return len(match.group(1)) if match else 0

def normalize_card_headers(text: str) -> str:
    """Убрать повторённое название карточки только из заголовков полей."""
    lines = text.splitlines(keepends=True)
    prefixes = set()

    for line in lines:
        match = re.fullmatch(
            r"(Баг\s*№\s*\d+\s*:\s*.+?\s+)Номер бага",
            line.strip(),
            flags=re.IGNORECASE,
        )
        if match:
            prefixes.add(match.group(1))

    if len(prefixes) != 1:
        return text

    prefix = prefixes.pop()
    normalized = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith(prefix):
            field = stripped[len(prefix):]
            if label(field) in FIELD_NAMES:
                ending = line[len(line.rstrip("\r\n")):]
                normalized.append(field + ending)
                continue

        normalized.append(line)

    return "".join(normalized)

def extract_field(text: str, field_name: str) -> str | None:
    """Извлечь единственное поле без пересказа его содержимого."""
    lines = normalize_card_headers(text).splitlines(keepends=True)
    starts = [
        index
        for index, line in enumerate(lines)
        if label(line) == field_name
    ]

    # Не угадываем, если одинаковый раздел встретился несколько раз.
    if len(starts) != 1:
        return None

    start = starts[0]
    start_level = heading_level(lines[start])
    end = len(lines)

    for index in range(start + 1, len(lines)):
        current_level = heading_level(lines[index])
        if (
            label(lines[index]) in FIELD_NAMES
            or (
                start_level
                and current_level
                and current_level <= start_level
            )
        ):
            end = index
            break

    return "".join(lines[start + 1:end]).strip()


def card_from_text(text: str, metadata: dict) -> dict:
    bug_id = str(metadata.get("bug_id", ""))
    declared_id = extract_field(text, "номер бага")

    if declared_id is not None and declared_id.strip() != bug_id:
        # Несогласованную карточку можно показать модели как документ,
        # но нельзя использовать для адресного выбора решения.
        bug_id = ""

    return {
        "bug_id": bug_id,
        "name": extract_field(text, "название") or f"Баг №{bug_id}",
        "solution": extract_field(text, "временное решение"),
        "source_hash": metadata.get("file_sha256", ""),
        "text": text,
    }


def relevant_ids(answer: str) -> list[str]:
    """Взять номера только из раздела 'Релевантные баги'."""
    result = []
    in_relevant_section = False

    for line in answer.splitlines():
        current = label(line)
        if current == "релевантные баги":
            in_relevant_section = True
            continue
        if current == "менее релевантные баги":
            in_relevant_section = False
            continue

        match = CARD_LINE.match(line)
        if in_relevant_section and match:
            bug_id = match.group(1)
            if bug_id not in result:
                result.append(bug_id)

    return result


def make_options(answer: str, source_cards: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for card in source_cards:
        grouped[card["bug_id"]].append(card)

    options = []

    for bug_id in relevant_ids(answer):
        cards = grouped.get(bug_id, [])
        if not cards:
            continue

        # При конфликтующих версиях одного бага выбор не предлагаем.
        versions = {
            (card["source_hash"], card["name"], card["solution"])
            for card in cards
        }
        if len(versions) != 1:
            continue

        card = cards[0]
        solution = card["solution"]

        if (
            not card["source_hash"]
            or not isinstance(solution, str)
            or solution.strip().casefold().rstrip(".") in EMPTY_SOLUTIONS
            or len(solution.encode("utf-16-le")) // 2 > 4096
        ):
            continue

        options.append(
            {
                "bug_id": bug_id,
                "name": card["name"],
                "solution": solution,
                "source_hash": card["source_hash"],
            }
        )

    return options