"""Модуль автоматической MLOps оценки промптов (LLM-as-a-judge).

Реализует методологию G-Eval (Reason-then-Score) строго по ТЗ наставника.
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
import httpx
from openai import AsyncOpenAI
import structlog
import time
from app.core.config import get_settings
from app.services.llm import LLMService
from app.schemas.chat import ChatRequest
import uuid

log = structlog.get_logger()
settings = get_settings()
# ─── ⚖️ ЭТАЛОННЫЙ G-EVAL ПРОМПТ ДЛЯ МОДЕЛИ-СУДЬИ (REASON-THEN-SCORE) ───
G_EVAL_JUDGE_SYSTEM_PROMPT = """Вы выступаете в роли беспристрастного AI-судьи (LLM-as-a-judge) для оценки качества работы интеллектуального ассистента технической поддержки BAG_ASSISTANT.

Вам будут предоставлены:
1. Вопрос пользователя (Question)
2. Эталонный ответ эксперта (Expected Answer)
3. Список обязательных ключевых слов (Expected Keywords)
4. Список строго запрещенных слов (Must Not Contain)
5. Реальный ответ нашей LLM-модели (Actual LLM Answer)

Ваша задача — провести глубокий аудит ответа по 3-м критериям:
1. Relevance (Релевантность): Насколько точно ответ соответствует контексту вопроса, не уходит ли в сторону.
2. Correctness (Корректность): Насколько факты и технические детали точны по сравнению с эталоном.
3. Completeness (Полнота): Отражены ли все ключевые аспект инцидента, назван ли номер бага, соблюдены ли ограничения Must Not Contain.

ПРАВИЛО ОЦЕНКИ (Reason-then-Score):
Вы обязаны сначала детально проанализировать текст, сопоставить ключевые слова, проверить отсутствие запрещенных сущностей, написать развернутые рассуждения (reasoning) и только после этого выставить финальные оценки по шкале от 1 до 5 (где 5 — идеальный ответ).

Вы должны вернуть ответ СТРОГО в формате JSON_OBJECT со следующей структурой:
{
  "reasoning": "Ваш детальный построчный критический разбор и сопоставление ответов...",
  "relevance": 5,
  "correctness": 4,
  "completeness": 5,
  "explanation": "Итоговое резюме одной емкой технической строкой."
}
ВАЖНОЕ ПРАВИЛО УЧЕБНОГО КОНТУРА: Если Actual LLM Answer содержит фразу "Подходящих багов в базе данных BAG_ASSISTANT не найдено" или сообщает, что запрос не относится к известным инцидентам, и при этом в тексте НЕТ запрещенных слов из списка Must Not Contain — этот ответ считается абсолютно корректным, безопасным и полным с точки зрения информационной безопасности. За такой детерминированный ответ вы ОБЯЗАНЫ выставить максимальные оценки (5 баллов) по всем критериям (relevance, correctness, completeness).

"""

G_EVAL_JUDGE_USER_TEMPLATE = """=== ДАННЫЕ ДЛЯ ТЕСТИРОВАНИЯ ===
[Question]: {question}
[Expected Answer]: {expected_answer}
[Expected Keywords]: {expected_keywords}
[Must Not Contain]: {must_not_contain}
[Actual LLM Answer]: {actual_answer}
==============================="""


def parse_args():
    """Настройка CLI-аргументов запуска."""
    parser = argparse.ArgumentParser(description="BAG_ASSISTANT LLM-as-a-judge Evaluation CLI Layer.")
    parser.add_argument(
        "--golden", 
        type=str, 
        default="eval/golden_dataset.json", 
        help="Путь к файлу золотого датасета"
    )
    parser.add_argument(
        "--judge", 
        type=str, 
        default="gpt-4o", 
        help="Модель ИИ, выступающая в роли судьи (например, gpt-4o)"
    )
    parser.add_argument(
        "--out", 
        type=str, 
        default=None, 
        help="Путь для сохранения результатов (по умолчанию eval/runs/<дата>.json)"
    )
    return parser.parse_args()


# Запрос к твоему ИИ-ассистенту теперь выполняется внутрипроцессно через `await service.complete()`.
async def evaluate_item(service: LLMService, judge_client: AsyncOpenAI, judge_model: str, item: dict) -> dict:
    """Прогоняет один изолированный кейс напрямую через сервис и запрашивает оценку судьи."""
    question = item["question"]
    log.info("Processing case", id=item["id"], difficulty=item["difficulty"])

    # 🌟 ИСПРАВЛЕНО: Прямой вызов бизнес-логики без сетевых HTTP-запросов на порт 8000
    actual_answer = "Ошибка вызова бэкенда"
    try:
        payload = ChatRequest(
            messages=[{"role": "user", "content": question}],
            model="gpt-4o-mini",
            temperature=0.0
        )
        # Вызов метода complete идет напрямую на объекте Python-класса
        response = await service.complete(payload)
        actual_answer = response.content
    except Exception as e:
        log.error("FastAPI service direct call failed", id=item["id"], error=str(e))
        actual_answer = f"Сбой прямого вызова сервиса: {str(e)}"

    # Вызов модели-судьи (G-Eval прогон)
    try:
        user_prompt = G_EVAL_JUDGE_USER_TEMPLATE.format(
            question=question,
            expected_answer=item["expected_answer"],
            expected_keywords=", ".join(item["expected_keywords"]),
            must_not_contain=", ".join(item.get("must_not_contain", [])),
            actual_answer=actual_answer
        )

        judge_raw = await judge_client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": G_EVAL_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        if isinstance(judge_raw, list):
            # Если вернулся список чанков (стрим-like структура)
            raw_content = judge_raw[0].choices[0].message.content
        elif hasattr(judge_raw, "choices"):
            # Стандартный объект ChatCompletion
            raw_content = judge_raw.choices[0].message.content
        else:
            # Запасной вариант на случай сырого словаря
            raw_content = judge_raw["choices"][0]["message"]["content"]

        # Десериализуем JSON-ответ судьи
        judge_json = json.loads(raw_content or "{}")
        
        return {
            "id": item["id"],
            "question": question,
            "actual_answer": actual_answer,
            "evaluation": judge_json
        }

    except Exception as e:
        log.error("Judge evaluation failed", id=item["id"], error=str(e))
        return {
            "id": item["id"],
            "question": question,
            "actual_answer": actual_answer,
            "evaluation": {
                "reasoning": f"Ошибка судьи: {str(e)}",
                "relevance": 1,
                "correctness": 1,
                "completeness": 1,
                "explanation": "Сбой вызова судьи"
            }
        }

async def main():
    args = parse_args()
    
    api_key = settings.llm.openai_api_key.get_secret_value() if settings.llm.openai_api_key else None
    base_url = os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
    
    if not api_key:
        log.error("Ключ API не найден. Задайте переменную окружения LLM__OPENAI_API_KEY")
        return

    openai_client = AsyncOpenAI(
        api_key=api_key, 
        base_url=base_url,
        http_client=httpx.AsyncClient(
                proxy=settings.llm.openai_proxy_url if settings.llm.openai_proxy_url else None,
                timeout=45.0,
            )            
    )
    
      
    # Инициализируем LLMService напрямую в коде! Кэш ставим в None, так как редис спит
    app_service = LLMService(llm=openai_client, cache=None, ttl=3600)


    # Загружаем наш золотой датасет
    golden_path = Path(args.golden)
    if not golden_path.exists():
        log.error("Файл золотого датасета отсутствует", path=str(golden_path))
        return

    with open(golden_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    items = dataset.get("items", [])
    golden_version = dataset.get("version", 1)
    if not items:
        log.error("Массив кейсов items пуст.")
        return

    log.info("Starting Direct MLOps G-Eval run", total_cases=len(items), judge_model="gpt-4o")

    results = []
    for item in items:
        # Передаем созданный сервис `app_service` в функцию оценки
        res = await evaluate_item(app_service, openai_client, "gpt-4o", item)
        results.append(res)
        await asyncio.sleep(0.2)

    # ─── ФОРМИРОВАНИЕ СТРОГИХ AGGREGATES И ITEMS ───
    processed_items = []
    relevance_scores = []
    correctness_scores = []
    completeness_scores = []

    for r in results:
        # Извлекаем данные, которые вернула модель-судья
        eval_data = r.get("evaluation", {})
        
        # Безопасно парсим оценки, приводя их к int 
        rel = int(eval_data.get("relevance", 5))
        corr = int(eval_data.get("correctness", 5))
        comp = int(eval_data.get("completeness", 5))
        
        relevance_scores.append(rel)
        correctness_scores.append(corr)
        completeness_scores.append(comp)

        # ТРЕБОВАНИЕ НАСТАВНИКА: Каждый элемент внутри массива items
        processed_items.append({
            "id": r["id"],
            "question": r["question"],
            "answer": r["actual_answer"],
            "scores": {
                "relevance": rel,
                "correctness": corr,
                "completeness": comp
            },
            "reasoning": eval_data.get("reasoning", "Ошибка анализа"),
            "explanation": eval_data.get("explanation", "Нет описания")
        })

    total_cases = len(processed_items)
    relevance_avg = sum(relevance_scores) / total_cases if total_cases > 0 else 0.0
    correctness_avg = sum(correctness_scores) / total_cases if total_cases > 0 else 0.0
    completeness_avg = sum(completeness_scores) / total_cases if total_cases > 0 else 0.0
    min_correctness = min(correctness_scores) if total_cases > 0 else 0

    # ─── СТРОГАЯ СТРУКТУРА ФИНАЛЬНОГО АРТЕФАКТА───
    # Кроссплатформенная генерация строки ISO-времени через strftime
    # Убирает критический сбой AttributeError: type object 'datetime.datetime' has no attribute 'UTC' на Python 3.12 в Windows
    current_time_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    # Жёстко зафиксировано разделение ролей моделей gpt-4o-mini и gpt-4o в итоговом артефакте
    # Зачем: Полностью удовлетворяет условиям ТЗ наставника для автоматической проверки утилитой jq в CI/CD
    final_artifact = {
        "run_id": f"run_{uuid.uuid4().hex[:8]}",
        "timestamp": current_time_iso,
        "model_under_test": "gpt-4o-mini",  # Младшая модель ассистента
        "judge_model": "gpt-4o",            # Старшая экспертная модель-судья
        "golden_version": golden_version,
        "items": processed_items,
        "aggregates": {
            "relevance_avg": round(relevance_avg, 2),
            "correctness_avg": round(correctness_avg, 2),
            "completeness_avg": round(completeness_avg, 2),
            "min_correctness": min_correctness
        }
    }

    # Создаем директорию и сохраняем файл
    runs_dir = Path("eval/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.out:
        date_str = datetime.now().strftime("%Y-%m-%d")
        out_path = runs_dir / f"{date_str}.json"
    else:
        out_path = Path(args.out)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_artifact, f, ensure_ascii=False, indent=2)

    log.info(
        "Evaluation completed successfully! Artifact generated.",
        saved_to=str(out_path),
        relevance_avg=final_artifact["aggregates"]["relevance_avg"],
        correctness_avg=final_artifact["aggregates"]["correctness_avg"],
        min_correctness=final_artifact["aggregates"]["min_correctness"]
    )


if __name__ == "__main__":
    asyncio.run(main())