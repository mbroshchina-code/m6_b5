"""Проверка состава активной коллекции без LLM и изменения данных."""

import hashlib
import json
from pathlib import Path

from qdrant_client import QdrantClient

from app.core.config import get_settings


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {110, 112}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    settings = get_settings()

    source = read_json(ROOT / "data/bugs_database.json")
    source_ids = {int(item["id"]) for item in source}

    store_path = (
        ROOT
        / settings.ingest_state_dir
        / settings.ingest_collection
        / "docstore.json"
    )
    store = read_json(store_path)["docstore/data"]

    found = set()
    inconsistent = set()
    unidentified = 0
    points_count = 0

    key = settings.qdrant_api_key
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=key.get_secret_value() if key else None,
        timeout=60,
        trust_env=False,
    )
    try:
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=settings.ingest_collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                points_count += 1
                payload = point.payload or {}
                raw_id = payload.get("bug_id")

                if not str(raw_id).isdigit():
                    unidentified += 1
                    continue

                bug_id = int(raw_id)
                found.add(bug_id)

                if bug_id not in REQUIRED:
                    continue

                document = store.get(payload.get("doc_id"), {})
                metadata = document.get("__data__", {}).get("metadata", {})
                stored_hash = metadata.get("file_sha256")
                relative_path = metadata.get("relative_path")

                if (
                    str(metadata.get("bug_id")) != str(bug_id)
                    or not stored_hash
                    or payload.get("file_sha256") != stored_hash
                    or not relative_path
                ):
                    inconsistent.add(bug_id)
                    continue

                original = ROOT / settings.ingest_data_root / relative_path
                if (
                    not original.is_file()
                    or hashlib.sha256(original.read_bytes()).hexdigest()
                    != stored_hash
                ):
                    inconsistent.add(bug_id)

            if offset is None:
                break
    finally:
        client.close()

    missing = source_ids - found

    print("Активная коллекция:", settings.ingest_collection)
    print("Багов в исходном JSON:", len(source_ids))
    print("Разных номеров багов в Qdrant:", len(found))
    print("Фрагментов в Qdrant:", points_count)
    print("Фрагментов без распознанного номера:", unidentified)
    print("Нет в коллекции из исходного JSON:", sorted(missing))
    print("Номера вне исходного JSON:", sorted(found - source_ids))

    for bug_id in sorted(REQUIRED):
        if bug_id not in found:
            status = "ОТСУТСТВУЕТ"
        elif bug_id in inconsistent:
            status = "НЕ СОВПАДАЮТ данные Qdrant, docstore или файла"
        else:
            status = "есть; версии Qdrant, docstore и файла совпадают"
        print(f"Баг {bug_id}: {status}")

    if REQUIRED <= found and not inconsistent:
        print("OK: необходимые карточки для пяти задач присутствуют.")
    else:
        print("СТОП: сначала нужно восстановить необходимые карточки.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Ошибка проверки:", type(exc).__name__)
        print("Сравнительный запуск пока не выполняйте.")
        raise SystemExit(1)