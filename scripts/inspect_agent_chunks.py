import json

from qdrant_client import QdrantClient, models

from app.core.config import get_settings


def main():
    settings = get_settings()
    key = settings.qdrant_api_key
    count = 0

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
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="bug_id",
                            match=models.MatchValue(value=142),
                        )
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                raw = payload.get("_node_content")
                if not raw:
                    raise RuntimeError("У фрагмента отсутствует _node_content")
                node = json.loads(raw) if isinstance(raw, str) else raw
                count += 1
                print(f"\n=== Фрагмент {count}; ID: {point.id} ===")
                print(node["text"])

            if offset is None:
                break
    finally:
        client.close()
        
    print(f"\nВсего фрагментов: {count}")


if __name__ == "__main__":
    main()