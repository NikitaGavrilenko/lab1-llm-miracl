# Лабораторная 1 — MIRACL ru

[Репозиторий проекта](https://github.com/NikitaGavrilenko/lab1-llm-miracl).

Итоговый отчёт: [Word](report/Lab1_MIRACL_final.docx), [PDF](report/Lab1_MIRACL_final.pdf). Данные: data/prepared/passport.json.
Сырые ответы и метрики: results/. Матрица 15 работ: report/literature_matrix.csv.

Восемь конкретных локальных LLM сравниваются в zero-shot и two-shot на 40 общих запросах / 416 judgments. Это оценка релевантности и условный reranking, не полнокорпусный retrieval. Отдельно выполнены последовательный поиск на 20 новых запросах по 41007 пассажам и настоящая LoRA малой Qwen0.5B.

Выбор только по validation: qwen3:14b, few-shot. Test не использован для выбора.

## Полное продолжение

```powershell
python src/finish_lab1.py
```

Единый скрипт использует кэш, последовательно запускает недостающие измерения, фиксирует выбор, проверяет метрики независимо, выполняет поиск и LoRA, собирает Word, экспортирует и рендерит страницы. Он сохраняет этапы в .workflow/full_run_state.json. Модели не запускаются одновременно. Платные API не используются.

## Отдельные этапы

```powershell
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
```

Основной Python на исходной машине D:/Anaconda3/envs/ml/python.exe; поиск также проверен на D:/Anaconda3/python.exe. Зависимости — requirements.txt; LoRA окружение и версии зафиксированы в results/peft_run_manifest.json. Word авторинг использует bundled python-docx; экспорт для проверки требует установленного Microsoft Word, PNG создаёт bundled Poppler.

Локальные веса Ollama и их полные digest — results/runtime.json. При воспроизведении тег может измениться: сверяйте digest. Контекст8192, temperature0, seed42, общий лимит384, JSON schema. Невалидный ответ — отказ без repair/retry/BM25 fallback. JSON p — заявленная уверенность, не token posterior. CI — bootstrap2000 по query_id.

Для нового независимого опыта используйте отдельную копию проекта с пустой results/ и без обученных adapters/; сохраните текущие результаты отдельно. Сначала выполните prepare_data.py. В конфигурации runner при переносе задайте пути к Python и bundled runtime. Ollama должен обслуживать модели из configs/protocol.json. Адаптер и базовые веса доступны отдельно; models/ и виртуальное окружение в архив не включены. Существующий selection никогда не пересчитывается по test.

Датасет закреплён по revisions и SHA256; все выбранные qrels сохранены. Тексты зеркала покрывают официальные train/dev judgments, 16 текстов сверены с официальным корпусом. Корпус ограничен исторически размеченными пассажами; qrels неполны. Результаты не сравниваются с полным MIRACL leaderboard. Карточки официальных данных указывают Apache-2.0; у зеркала отдельная лицензия не указана, происхождение MIRACL/Wikipedia раскрыто.

Оригиналы прежней работы сохранены в previous/. Старый корневой Word с незаполненными таблицами не является итоговым отчётом. Пилоты и несовместимость Qwen3-VL сохранены в results/pilot и исключены из основного сравнения. Размеры моделей и квантования различаются; это системный локальный benchmark. LoRA — отдельный контроль на0.5B, не QLoRA и не девятая строка.

Автоматические проверки — results/independent_numeric_checks.json и results/final_audit.json; независимая критика модели — results/independent_model_review.json. Все 19 страниц проверены визуально; таблицы Word совпадают с экспортом, встроенные графики — с исходными PNG. Независимый аудит сырых результатов: results/final_independent_audit.json. Пройдено 14 unit-тестов и 12 проверок поиска. Репозиторий опубликован: https://github.com/NikitaGavrilenko/lab1-llm-miracl.


## Лабораторная работа 2

[Мультимодальный финансовый прогноз](lab2-multimodal/README.md): восемь схем объединения текста, числового ряда и графика, реальные эксперименты, абляции и статистические проверки. [Итоговый Word-отчёт](lab2-multimodal/report_lab2.docx). Полный запуск: `cd lab2-multimodal` и `python run_all.py`. Преимущество fusion не подтверждено; отрицательный результат сохранён.
