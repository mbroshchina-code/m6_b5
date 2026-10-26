import json

from app.services.agent_react_runtime import Worker


def main():
    worker = Worker()

    try:
        with worker:
            print("STARTUP OK")
    except Exception as exc:
        print("STARTUP FAILED:", type(exc).__name__)
    finally:
        print(
            json.dumps(
                worker.startup,
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()