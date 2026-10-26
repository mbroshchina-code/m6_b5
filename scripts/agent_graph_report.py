"""Формирование отчёта по сохранённым результатам бенчмарка."""

import json
from pathlib import Path
from statistics import mean
from scripts.check_graph_results import check_result


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATIONS = ("b62", "custom", "prebuilt")
FIELDS = (
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "total_steps",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def average(rows, field):
    values = [row.get(field) for row in rows]
    if not values or any(value is None for value in values):
        return None
    return mean(values)


def number(value):
    return "—" if value is None else f"{value:.2f}"


def make_report(folder):
    metadata = read_json(folder / "metadata.json")
    tasks = read_json(folder / "tasks.json")
    rows = [
        read_json(path)
        for path in sorted(folder.glob("result-*.json"))
    ]

    mermaid = (ROOT / "docs/agent-graph-custom.mmd").read_text(
        encoding="utf-8"
    ).strip()

    lines = [
        "# LangGraph: сравнение трёх реализаций",
        "",
        "## 1. Конфигурация",
        "",
        f"Начало UTC: {metadata['started_utc']}.",
        f"Сохранено попыток: {len(rows)} из 45.",
        f"Модель: `{metadata['model']}`; "
        f"после REVISE: `{metadata['premium_model']}`.",
        "Температура не передаётся; reasoning_effort=none.",
        "Б6.2: исходный agent_react.py, max_iterations=10, "
        "таймаут итерации 15 секунд.",
        "Custom и prebuilt: максимум 6 вызовов основной модели, "
        "включая исправления финального ответа.",
        "Максимум ревизий: 2. Вызовы критика считаются отдельно.",
        "Tools: search_knowledge_base, get_current_time, "
        "send_telegram_message. Отправка учебная.",
        "",
        "Версии: " + ", ".join(
            f"{name}={value}"
            for name, value in metadata["versions"].items()
        ),
        "",
        "Каждая попытка выполняется в новом процессе, последовательно. "
        "latency_ms измеряется через perf_counter в родительском процессе "
        "и включает запуск Python, импорты, подготовку, поиск, модель, "
        "критика и завершение процесса. Прогрева нет.",
        "Порядок реализаций меняется между повторами. "
        "Это сравнение полного запуска, а не чистых накладных расходов графа.",
        "Токены включают основную модель, исправления и критика. "
        "Токены эмбеддингов не включаются.",
        "total_steps — число вызовов основной модели; критик в него не входит.",
        "Конфигурация поиска и отпечатки исходников сохранены в metadata.json.",
        "",
        "## 2. State contract",
        "",
        "- messages: история с reducer add_messages.",
        "- iteration_count: число вызовов основной модели, replace.",
        "- tool_results: накопление результатов через operator.add.",
        "- revisions: число разрешённых ревизий, replace.",
        "- premium_next: выбор модели следующего шага, replace.",
        "- stop_reason: причина принудительной остановки, replace.",
        "- llm_calls: накопление статистики модели и критика через operator.add.",
        "",
        "В состоянии нет SDK-клиентов, сессий и ключей. "
        "Сообщения поддерживаются сериализатором LangGraph.",
        "",
        "## 3. Router и условия остановки",
        "",
        "Router сначала проверяет iteration_count >= 6. "
        "Затем направляет вызовы tools в execute_tool; "
        "ответ без tool calls — в force_finish.",
        "На шестом вызове новые tools не исполняются: "
        "создаются ToolMessage об отмене и понятный итоговый ответ.",
        "Критик может остановить выполнение при неверном ответе "
        "или исчерпании двух ревизий. Финальное исправление "
        "использует оставшийся лимит вызовов основной модели.",
        "",
        "## 4. Схема custom",
        "",
        "```mermaid",
        mermaid,
        "```",
        "",
        "Схема prebuilt сохранена в agent-graph-prebuilt.mmd.",
        "",
        "## 5. Бенчмарк",
        "",
        "Средние рассчитаны по сохранённым попыткам, включая остановки. "
        "Для законченного эксперимента в каждой строке нужно n=3. "
        "«Без остановки» не означает проверенную правильность ответа.",
        "Если статистика токенов хотя бы одной попытки неизвестна, "
        "среднее обозначается «—», а не нулём.",
        "",
        "| Задача | Реализация | n | Без остановки | latency_ms | "
        "prompt_tokens | completion_tokens | total_steps |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for task in tasks:
        for implementation in IMPLEMENTATIONS:
            group = [
                row for row in rows
                if row["task_id"] == task["id"]
                and row["implementation"] == implementation
            ]
            completed = sum(
                not row.get("error") and not row.get("tool_error")
                for row in group
            )
            values = " | ".join(
                number(average(group, field)) for field in FIELDS
            )
            lines.append(
                f"| {task['id']} | {implementation} | {len(group)} "
                f"| {completed} | {values} |"
            )

    lines += [
        "",
        "## 6. Custom vs prebuilt",
        "",
        "В custom вручную написаны state, вызов модели, исполнение tools, "
        "router, рёбра и force-finish. В prebuilt модельный цикл "
        "и исполнение tools собирает create_agent; ограничения, "
        "обработку ошибок и self-reflection добавляют middleware.",
        "Custom удобнее при необходимости явно управлять переходами. "
        "Prebuilt удобнее для стандартного цикла с расширениями через middleware.",
        "",
    ]
    
    lines += [
        "Проверки формата и последовательности действий:",
        "",
        "| Реализация | Прошли проверки | Проверено попыток |",
        "|---|---:|---:|",
    ]

    failures = []

    for implementation in IMPLEMENTATIONS:
        group = [
            row for row in rows
            if row["implementation"] == implementation
        ]
        passed = 0

        for row in group:
            errors = check_result(row)
            if not errors:
                passed += 1
            else:
                label = (
                    f"{row['task_id']} / {implementation} / "
                    f"повтор {row['repeat']}"
                )
                failures.append(
                    f"- {label}: {'; '.join(errors)}."
                )

        lines.append(
            f"| {implementation} | {passed} | {len(group)} |"
        )

    lines += [
        "",
        "Для Б6.2 проверяется прежний учебный формат, для графов — "
        "формат основного промпта дипломного ассистента. "
        "Эти проверки не заменяют ручную проверку достоверности.",
        "",
    ]

    if failures:
        lines += [
            "Найденные нарушения:",
            "",
            *failures,
            "",
        ]
        
    expected_runs = {
        (task["id"], implementation, repeat)
        for task in tasks
        for implementation in IMPLEMENTATIONS
        for repeat in range(1, 4)
    }
    actual_runs = {
        (row["task_id"], row["implementation"], row["repeat"])
        for row in rows
    }
    complete = (
        len(rows) == len(expected_runs)
        and actual_runs == expected_runs
        and all(row.get("usage_complete") for row in rows)
    )
    
    if complete:
        lines.append(
            "Все 45 попыток сохранены; учёт токенов полный. "
            "Средние включают остановки и неправильные ответы, "
            "поэтому скорость нужно оценивать вместе с качеством."
        )
        
        for implementation in IMPLEMENTATIONS:
            group = [
                row for row in rows
                if row["implementation"] == implementation
            ]
            lines.append(
                f"- {implementation}: среднее время "
                f"{average(group, 'latency_ms'):.2f} мс; "
                f"вход {average(group, 'prompt_tokens'):.2f}; "
                f"выход {average(group, 'completion_tokens'):.2f} токенов."
            )

        custom = average(
            [r for r in rows if r["implementation"] == "custom"],
            "latency_ms",
        )
        prebuilt = average(
            [r for r in rows if r["implementation"] == "prebuilt"],
            "latency_ms",
        )
        lines += [
            "",
            f"В этом эксперименте разница prebuilt − custom составляет "
            f"{prebuilt - custom:.2f} мс в среднем на задачу.",
            "Вывод предварительный до ручной проверки ответов. "
            "Пять задач и три повтора не доказывают общего превосходства.",
        ]
    else:
        lines.append(
            "Итоговое сравнение отложено: набор из 45 попыток "
            "неполон либо отсутствует статистика токенов. "
            "Доступные результаты приведены в таблице."
        )

    lines += [
        "",
        "Ограничения сравнения с Б6.2: у исходного агента лимит 10, "
        "у графов 6; различаются организация подготовки и таймауты. "
        "В графах финальная ревизия не вызывает новые tools. "
        "Поэтому разницу нельзя объяснять только переходом на LangGraph.",
        "Дополнительное ограничение: графы используют обновлённые "
        "правила дипломного ассистента и критика, а Б6.2 сохраняет "
        "прежние промпты. Это также влияет на поведение и токены.",
        "",
        "## 7. Найденный дефект",
        "",
        "При чтении кода перед бенчмарком обнаружено: "
        "_finish_update возвращал сообщение о лимите, "
        "но stop_reason оставался None. Это позволяло ошибочно "
        "считать остановку успешным завершением.",
        "Исправление: stop_reason=max_iterations; для пустого ответа — "
        "empty_answer. В check_graph_offline.py добавлена проверка "
        "stop_reason для обоих графов при достижении лимита.",
        "",
        "В эксперименте 20261018_114929_118213 найдено ложное "
        "подтверждение отправки: custom сообщил об успехе после "
        "одного поиска, а критик одобрил ответ. Добавлена проверка "
        "предусмотренного шаблона подтверждения по tool_results: "
        "необходим успешный результат send_telegram_message "
        "для соответствующего chat_id.",
        "",
        "В том же эксперименте критик выдавал противоречивые "
        "замечания на этапе observation. Уточнён предмет проверки: "
        "последний выполненный инструмент, а не отсутствующий "
        "финальный ответ. В payload добавлен last_tool_result.",
        "",
        "Офлайн-проверки подтверждают блокировку ложного подтверждения. "
        "Влияние изменений на выполнение задач оценивается отдельно "
        "по результатам реальной модели.",
        "",
        "## 8. Переход к checkpointing",
        "",
        "Обе фабрики принимают checkpointer. В вызов передаётся thread_id "
        "с идентификаторами задачи, реализации и повтора. "
        "Без checkpointer он не обеспечивает сохранение состояния.",
        "Далее нужны saver, управление его async-жизненным циклом, "
        "правила продолжения существующего thread и идемпотентность "
        "инструментов перед подключением interrupt.",
        "При продолжении thread нельзя повторно передавать полную "
        "старую историю и сбрасывать счётчики как для нового запуска.",
        "",
        "## Ответы для ручной проверки",
        "",
    ]

    for task in tasks:
        lines += [
            f"### {task['id']}: {task['title']}",
            "",
            f"Ожидание: {task['expected']}",
            "",
        ]
        for row in rows:
            if row["task_id"] != task["id"]:
                continue
            answer = str(row.get("answer", "")).replace("```", "'''")
            lines += [
                f"**{row['implementation']}, повтор {row['repeat']}**",
                "",
                f"Остановка: {row.get('error') or 'нет'}.",
                "",
                "```text",
                answer,
                "```",
                "",
            ]

    text = "\n".join(lines) + "\n"
    (folder / "report.md").write_text(text, encoding="utf-8")
    (ROOT / "docs/agent-graph-report.md").write_text(text, encoding="utf-8")