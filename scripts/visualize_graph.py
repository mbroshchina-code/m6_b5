"""Сохранить Mermaid-схемы без обращения к внешним сервисам."""

from pathlib import Path

from app.services.agent_graph import custom_graph, prebuilt_graph


def main():
    docs = Path(__file__).resolve().parents[1] / "docs"
    docs.mkdir(parents=True, exist_ok=True)

    graphs = {
        "agent-graph-custom.mmd": custom_graph,
        "agent-graph-prebuilt.mmd": prebuilt_graph,
    }

    for filename, graph in graphs.items():
        target = docs / filename
        target.write_text(
            graph.get_graph().draw_mermaid() + "\n",
            encoding="utf-8",
        )
        print(f"Saved: docs/{filename}")


if __name__ == "__main__":
    main()