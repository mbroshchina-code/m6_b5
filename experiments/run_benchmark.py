from experiments.common import run_suite
from experiments.multi_agent_langgraph import build as build_multi
from experiments.single_agent_baseline import build as build_single


if __name__ == "__main__":
    run_suite(
        {
            "single": build_single,
            "multi": build_multi,
        }
    )