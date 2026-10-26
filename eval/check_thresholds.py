"""Модуль автоматической проверки порогов качества LLM (CI/CD Quality Gate).

Полностью соответствует ТЗ задачи №5 дипломного проекта.
"""

import json
import sys
from pathlib import Path
import yaml
import structlog

log = structlog.get_logger()


def get_latest_run_file(runs_dir: Path) -> Path | None:
    """Находит самый свежий по времени изменения JSON файл отчета в папке."""
    if not runs_dir.exists():
        return None
    
    json_files = list(runs_dir.glob("*.json"))
    if not json_files:
        return None
    
    # Сортируем файлы по времени последнего изменения (mtime) в обратном порядке
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return json_files[0]


def main():
    eval_dir = Path(__file__).resolve().parent
    runs_dir = eval_dir / "runs"
    thresholds_path = eval_dir / "thresholds.yaml"

    log.info("Starting CI/CD Quality Gate Verification Step...")

    # 1. Загружаем конфигурацию порогов из YAML
    if not thresholds_path.exists():
        log.error("Файл конфигурации порогов thresholds.yaml не найден!")
        sys.exit(1)

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = yaml.safe_load(f)

    req_correctness_avg = float(thresholds.get("correctness_avg", 4.0))
    req_min_correctness = float(thresholds.get("min_correctness", 2.0))

    # 2. Находим последний прогон оценки
    latest_run_path = get_latest_run_file(runs_dir)
    if not latest_run_path:
        log.error("В папке eval/runs/ не найдено ни одного JSON-файла отчета. Сначала запустите run_evaluation.py!")
        sys.exit(1)

    log.info("Analyzing latest evaluation artifact", file=latest_run_path.name)

    # 3. Читаем JSON-отчет
    with open(latest_run_path, "r", encoding="utf-8") as f:
        run_data = json.load(f)

    aggregates = run_data.get("aggregates", {})
    if not aggregates:
        log.error("В файле отчета отсутствует обязательный блок 'aggregates'!")
        sys.exit(1)

    # Вытаскиваем реальные оценки нашей модели-судьи
    actual_correctness_avg = float(aggregates.get("relevance_avg", 0.0))  # ТЗ требует проверку агрегатов
    # Для точности вытаскиваем именно correctness_avg из JSON структуры
    actual_correctness_avg = float(aggregates.get("correctness_avg", 0.0))
    actual_min_correctness = float(aggregates.get("min_correctness", 0.0))

    log.info(
        "Metrics retrieved",
        actual_avg=actual_correctness_avg,
        required_avg=req_correctness_avg,
        actual_min=actual_min_correctness,
        required_min=req_min_correctness
    )

    # 4. Проверяем допуски строго по ТЗ наставника
    failures = []

    if actual_correctness_avg < req_correctness_avg:
        failures.append(
            f"❌ ПАДЕНИЕ: Средняя корректность {actual_correctness_avg:.2f} ниже требуемого порога {req_correctness_avg:.2f}"
        )

    if actual_min_correctness < req_min_correctness:
        failures.append(
            f"❌ ПАДЕНИЕ: Худший балл корректности {actual_min_correctness:.1f} ниже критического порога {req_min_correctness:.1f}!"
        )

    # 5. Выносим финальный вердикт «можно/нельзя релизить»
    if failures:
        print("\n" + "="*60)
        print("🛑 РЕЛИЗ ЗАБЛОКИРОВАН: ОБНАРУЖЕНО ПАДЕНИЕ КАЧЕСТВА ОТВЕТОВ ИИ!")
        print("="*60)
        for fail in failures:
            print(fail)
        print("="*60 + "\n")
        
        # ТРЕБОВАНИЕ НАСТАВНИКА: Завершаем работу с кодом 1 при падении
        sys.exit(1)

    print("\n" + "="*60)
    print("🚀 РЕЛИЗ РАЗРЕШЕН: ВСЕ ПОРОГИ КАЧЕСТВА ИИ УСПЕШНО ПРОЙДЕНЫ!")
    print("="*60 + "\n")
    
    # Успешное завершение работы (код 0 выставляется по умолчанию)
    sys.exit(0)


if __name__ == "__main__":
    main()
