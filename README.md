# Лабораторная 1 — MIRACL ru

Итоговый отчёт: report/Lab1_MIRACL_final.docx. Данные: data/prepared/passport.json.
Сырые ответы и метрики: results/. Матрица 15 работ: report/literature_matrix.csv.

Восемь конкретных локальных LLM сравниваются в zero-shot и two-shot на 40 общих запросах / 416 judgments. Это оценка релевантности и условный reranking, не полнокорпусный retrieval. Отдельно выполнены последовательный поиск на 20 новых запросах по 41007 пассажам и настоящая LoRA малой Qwen0.5B.

Выбор только по validation: qwen3:14b, few-shot. Test не использован для выбора.

## Полное продолжение

python src/finish_lab1.py

Единый скрипт использует кэш, последовательно запускает недостающие измерения, фиксирует выбор, проверяет метрики независимо, выполняет поиск и LoRA, собирает Word, экспортирует и рендерит страницы. Он сохраняет этапы в .workflow/full_run_state.json. Модели не запускаются одновременно. Платные API не используются.

## Отдельные этапы

python src/prepare_data.py
python src/run_experiment.py --splits validation
python src/analyze_results.py --select
python src/run_experiment.py --splits test
python src/run_experiment.py --models qwen3:14b --splits application
python src/search_experiment.py
.venv-peft/Scripts/python.exe src/peft_experiment.py
python src/analyze_results.py
python -m unittest discover -s tests -v
python src/build_report.py

Основной Python на исходной машине D:/Anaconda3/envs/ml/python.exe; поиск также проверен на D:/Anaconda3/python.exe. Зависимости — requirements.txt; LoRA окружение и версии зафиксированы в results/peft_run_manifest.json. Word авторинг использует bundled python-docx; экспорт для проверки требует установленного Microsoft Word, PNG создаёт bundled Poppler.

Локальные веса Ollama и их полные digest — results/runtime.json. При воспроизведении тег может измениться: сверяйте digest. Контекст8192, temperature0, seed42, общий лимит384, JSON schema. Невалидный ответ — отказ без repair/retry/BM25 fallback. JSON p — заявленная уверенность, не token posterior. CI — bootstrap2000 по query_id.

Для нового независимого опыта скопируйте проект без predictions.jsonl, runtime.json, selection.json и search_trajectories.jsonl; сохраните текущие результаты отдельно. Существующий selection никогда не пересчитывается по test.

Датасет закреплён по revisions и SHA256; все выбранные qrels сохранены. Тексты зеркала покрывают официальные train/dev judgments, 16 текстов сверены с официальным корпусом. Корпус ограничен исторически размеченными пассажами; qrels неполны. Результаты не сравниваются с полным MIRACL leaderboard. Карточки официальных данных указывают Apache-2.0; у зеркала отдельная лицензия не указана, происхождение MIRACL/Wikipedia раскрыто.

Оригиналы прежней работы сохранены в previous/. Старый корневой Word с незаполненными таблицами не является итоговым отчётом. Пилоты и несовместимость Qwen3-VL сохранены в results/pilot и исключены из основного сравнения. Размеры моделей и квантования различаются; это системный локальный benchmark. LoRA — отдельный контроль на0.5B, не QLoRA и не девятая строка.

Автоматические проверки — results/independent_numeric_checks.json и results/final_audit.json; независимая критика модели — results/independent_model_review.json. Визуальная проверка всех PNG остаётся отдельным явно отмеченным этапом, пока первичный агент её не подтвердит. Удалённый репозиторий не настроен; публичная ссылка не выдумывается.
