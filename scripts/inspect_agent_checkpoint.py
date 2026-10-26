import asyncio
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.agent_persistent_cards import relevant_ids


async def main():
    load_dotenv(ROOT / ".env")

    database = Path(
        os.getenv("AGENT_SQLITE_PATH", "var/agent/workarounds.sqlite3")
    )
    if not database.is_absolute():
        database = ROOT / database

    if not database.exists():
        print("Файл checkpoint не найден:", database)
        return

    config = {"configurable": {"thread_id": sys.argv[1]}}

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        saved = await saver.aget_tuple(config)

        if saved is None:
            print("Для этого thread_id checkpoint не найден.")
            return

        state = saved.checkpoint["channel_values"]

        print("delivery_status:", state.get("delivery_status"))
        print(
            "Распознанные номера:",
            relevant_ids(state.get("operator_answer", "")),
        )

        cards = state.get("source_cards", [])
        print("Количество исходных карточек:", len(cards))

        for index, card in enumerate(cards, 1):
            print(f"\nКАРТОЧКА {index}")
            for field in ("bug_id", "name", "solution", "source_hash"):
                print(f"{field}: {card.get(field)!r}")
            print("Начало исходного текста:")
            print(card.get("text", "")[:1500])


if __name__ == "__main__":
    asyncio.run(main())