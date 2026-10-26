"""Удаление пустых разделов из готового ответа о багах."""

import re

from app.core.rag_prompts import FALLBACK_ANSWER


# Начало карточки: * **162** — Название
# Также допускаем обычный номер, скобки и другой маркер списка.
CARD_START = re.compile(
    r"^[ \t]*(?:[-*+]|\d+[.)])?[ \t]*"
    r"(?:\*\*)?\[?\d+\]?(?:\*\*)?"
    r"[ \t]*[—–-][ \t]*\S",
    re.MULTILINE,
)

SECTION_NAMES = {
    "релевантные баги",
    "менее релевантные баги",
}


def is_section_heading(line: str) -> bool:
    plain = line.strip().strip("#*_: \t")
    return plain.casefold() in SECTION_NAMES


def normalize_bug_answer(text: str) -> str:
    if not text.strip():
        raise RuntimeError("Модель вернула пустой ответ")

    sections = []
    current = []

    for line in text.splitlines():
        if is_section_heading(line):
            if current:
                sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current))

    # Оставляем только разделы, содержащие начало карточки.
    kept = [
        section.strip()
        for section in sections
        if CARD_START.search(section)
    ]

    return "\n\n".join(kept) if kept else FALLBACK_ANSWER