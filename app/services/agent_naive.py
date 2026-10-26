import argparse
import json
import logging
import time

from app.services.agent_naive_tools import DISPATCH, TOOLS, make_client

from app.services.agent_naive_prompt import AGENT_RULES

def run_agent(task: str, max_steps: int = 6) -> dict:
    trace, steps, started = [], 0, time.perf_counter()
    row = {"step": 0, "tool_name": None, "tool_args": None, "tool_result": "",
           "llm_input_tokens": 0, "llm_output_tokens": 0, "duration_ms": 0}
    try:
        if not task.strip() or max_steps < 1:
            raise ValueError("Нужны непустая задача и max_steps >= 1")
        messages = [{"role": "system", "content": AGENT_RULES},
                    {"role": "user", "content": task}]
        with make_client() as client:
            for step in range(max_steps):
                steps, started = step + 1, time.perf_counter()
                row = dict(row, step=steps, tool_name=None, tool_args=None,
                           tool_result="", llm_input_tokens=0, llm_output_tokens=0)
                response = client.chat.completions.create(
                    model="gpt-5.4-mini", messages=messages, tools=TOOLS,
                    parallel_tool_calls=False, reasoning_effort="none")
                message = response.choices[0].message
                messages.append(message)
                if response.usage:
                    row.update(llm_input_tokens=response.usage.prompt_tokens,
                               llm_output_tokens=response.usage.completion_tokens)
                for call in message.tool_calls or []:
                    name, raw = call.function.name, call.function.arguments
                    row.update(tool_name=name, tool_args=raw)
                    try:
                        args = json.loads(raw)
                        result = DISPATCH[name](**args) if name in DISPATCH else f"Ошибка: неизвестный tool {name}"
                    except Exception as exc:
                        logging.exception("Ошибка инструмента %s", name)
                        result = f"Ошибка инструмента: {type(exc).__name__}"
                    row["tool_result"] = str(result)[:200]
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
                row["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
                trace.append(row.copy())
                logging.info("agent_step %s", json.dumps(row, ensure_ascii=False))
                if not message.tool_calls:
                    answer = (message.content or message.refusal or "").strip()
                    searched = any(
                        item["tool_name"] == "search_knowledge_base"
                        for item in trace
                    )
                    if "Подходящих багов не найдено" in answer and not searched:
                        return {
                            "answer": "Остановка: вывод об отсутствии багов сделан без поиска",
                            "steps": steps,
                            "trace": trace,
                            "error": "search_not_performed",
                        }
                    if not answer:
                        return {"answer": "Остановка: модель вернула пустой ответ",
                                "steps": steps, "trace": trace, "error": "empty_answer"}
                    return {"answer": answer, "steps": steps, "trace": trace}
        return {"answer": "Достигнут лимит шагов", "steps": steps, "trace": trace, "error": "max_steps"}
    except Exception as exc:
        row.update(tool_result=type(exc).__name__,
                   duration_ms=round((time.perf_counter() - started) * 1000, 2))
        trace.append(row.copy())
        logging.info("agent_error %s", json.dumps(row, ensure_ascii=False))
        return {"answer": f"Остановка: {type(exc).__name__}", "steps": steps,
                "trace": trace, "error": type(exc).__name__}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_agent(args.task, args.max_steps)
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.trace else result["answer"])