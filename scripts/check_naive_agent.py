from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

from app.services.agent_naive import run_agent


def response(message):
    return NS(
        choices=[NS(message=message)],
        usage=NS(prompt_tokens=10, completion_tokens=5),
    )


unknown_call = NS(
    id="test-call-1",
    function=NS(name="get_user_balance", arguments='{"user_id": "12345"}'),
)
tool_request = response(
    NS(tool_calls=[unknown_call], content=None, refusal=None)
)
final_response = response(
    NS(tool_calls=None, content="Инструмент недоступен.", refusal=None)
)

client = MagicMock()
client.__enter__.return_value = client
client.chat.completions.create.side_effect = [tool_request, final_response]

with patch("app.services.agent_naive.make_client", return_value=client):
    result = run_agent("Проверь баланс пользователя")

assert result["steps"] == 2, result
assert "error" not in result, result
assert result["trace"][0]["tool_name"] == "get_user_balance"
assert "Ошибка" in result["trace"][0]["tool_result"]

messages = client.chat.completions.create.call_args.kwargs["messages"]
tool_reply = next(m for m in messages if isinstance(m, dict) and m["role"] == "tool")
assert tool_reply["tool_call_id"] == "test-call-1"
assert "get_user_balance" in tool_reply["content"]
print("OK: неизвестный инструмент возвращён модели как ошибка; цикл продолжился.")

client.chat.completions.create.reset_mock()
client.chat.completions.create.side_effect = [tool_request]

with patch("app.services.agent_naive.make_client", return_value=client):
    result = run_agent("Проверь баланс пользователя", max_steps=1)

assert result["steps"] == 1, result
assert result["error"] == "max_steps", result
assert client.chat.completions.create.call_count == 1
print("OK: лимит шагов остановил цикл.")