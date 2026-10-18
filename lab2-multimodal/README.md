# Лабораторная № 2 — мультимодальный финансовый прогноз

Воспроизводимый CPU-эксперимент: восемь схем T/N/I и три отдельно переобученные попарные абляции. Цель — знак доходности следующей торговой сессии по заголовкам и прошлому OHLCV. Это компактные исследовательские baseline-модели, без обучения большой мультимодальной LLM.

**Результат отрицательный:** устойчивое преимущество fusion над выбранной по validation числовой моделью не подтверждено. `late_weighted` выбрала веса [0, 1, 0] и повторяет `numeric_only`; их test macro-F1 ≈ 0.500. На test `image_only` имеет macro-F1 ≈ 0.512, но выбирать модель по test нельзя. Проверки доходности стратегии, комиссий и исполнимости сделок не проводились; прибыльность не заявляется.

## Единый запуск

В чистом окружении после установки зависимостей: python run_all.py. Скрипт последовательно выполняет все этапы, останавливается при ошибке, сохраняет логи в results/logs и итог в results/pipeline_status.json. Для отдельного Python с python-docx используйте --report-python.

## Запуск

Нужны Python **3.10** (эксперимент: 3.10.20), CPU и интернет для первой загрузки пакетов/данных. GPU не требуется. В чистом окружении из корня проекта:

```text
python -m pip install -r requirements.txt
python src/download.py
python src/prepare.py
python src/run.py
python src/plots.py
python src/build_report.py
```

На этой машине интерпретатор 3.10: `D:\Anaconda3\envs\ml\python.exe`. Для изоляции можно создать `.venv` этим интерпретатором и выполнять команды через `.venv\Scripts\python.exe`, без активации PowerShell. Зафиксированы прямые зависимости и CPU PyTorch. Версии измеренного запуска сохранены в `results/run_summary.json`; длительность и latency зависят от оборудования. Исходные скрипты подготовки, обучения и построения графиков используют один Python; для отчёта требуется `python-docx`.

`download.py` читает `data/raw/kaggle_manifest.json`, использует URL с номером версии и проверяет SHA-256. При уже проверенных исходниках сеть не используется. Отсутствующий исходник восстанавливается из корректного локального ZIP либо скачивается. Совпадения имён внутри ZIP ограничены явно разрешёнными файлами; произвольные пути не извлекаются. Существующий файл с несовпавшей суммой сохраняется, выполнение останавливается. Для его явной замены:

```text
python src/download.py --force
```

Новая копия всё равно должна пройти manifest SHA-256; `--force` не разрешает принять другой dataset. При изменении серверного архива, блокировке Kaggle или требовании авторизации скрипт выдаст ошибку, а не подменит входные данные. После подготовки/обучения таблицы и результаты перезаписываются; для сравнения экспериментов предварительно сохраните отдельную копию.

## Данные и время

Источники: [Financial News with Ticker-Level Sentiment](https://www.kaggle.com/datasets/rdolphin/financial-news-with-ticker-level-sentiment), версия 1; [S&P500 Historical Data](https://www.kaggle.com/datasets/jacksaleeby/s-and-p500-historical-data), версия 1. Фактически JSON содержит 5548 новостей; используются только заголовки, а не готовые sentiment labels. Ссылки, размеры, checksums и заявленные загрузчиками лицензии находятся в manifest. Заявленная лицензия загрузчика не подтверждает права издателей и поставщика котировок.

Объект — тикер × опорная сессия XNYS. Новости привязываются к первому закрытию NYSE не раньше `published_utc`; после закрытия, в выходные и праздники — к следующей сессии. К моменту закрытия разрешены текущая и прошлые котировки. Цель: `AdjClose(next_session) / AdjClose(anchor_session) - 1 > 0`. Корректный timestamp отражает заявленное время публикации; архив не доказывает историческую доступность именно этой версии заголовка у поставщика.

| Часть | Опорные даты после очистки | Объектов |
|---|---|---:|
| train | 03.01–28.07.2023 | 2275 |
| validation | 01.08–28.09.2023 | 689 |
| test | 02.10–29.11.2023 | 688 |
| application | 01.12.2023–02.01.2024 | 273 |

Всего 3925 объектов, 453 тикера. Последняя новость датирована 31.12.2023, переносится на 02.01.2024; её target — **03.01.2024**. Переходы target через границы train/validation/test/application исключены. Нет календарного random split.

T: TF-IDF unigram/bigram → SVD, 64 признака. N: 20 сессий × 5 относительных OHLC/volume признаков, 100 чисел. I: свечной график того же окна, RGB 96×96 → 32×32 → PCA, 64 признака. I — производное представление рыночного ряда; его эффект нельзя называть добавлением независимой информации. Для расчёта относительных N требуется 21 строка истории. TF-IDF/SVD/PCA и StandardScaler обучаются только на train.

Ограничения: небольшой архив заголовков, отсутствие новостей по многим тикерам, selection bias и survivorship bias списка акций, сформированного в 2026 году. Результаты не являются point-in-time backtest всего исторического S&P500. Относительные признаки и target используют adjusted prices; инвариантность к общей мультипликативной корректировке проверяется в подготовке, но архив не восстанавливает все исторические revisions поставщика.

## Схемы и выбор параметров

| Схема | Реализация |
|---|---|
| `text_only` | LogisticRegression на T |
| `numeric_only` | LogisticRegression на N |
| `image_only` | LogisticRegression на I |
| `early_fusion` | Конкатенация T/N/I признаков → LogisticRegression |
| `late_mean` | Среднее вероятностей трёх unimodal моделей |
| `late_weighted` | Validation-подбор вероятностных весов на simplex с шагом 0.1, включая вершины |
| `intermediate_mlp` | Проекции модальностей по 16 → concat → MLP 32 |
| `attention_fusion` | Проекции → learned-query pooling трёх modality tokens → MLP 32 |

`attention_fusion` — attention pooling, не pairwise cross-attention и не реализация MulT. Название `early_fusion` здесь означает объединение фиксированных признаков после кодирования. Дополнительно с нуля обучаются TN/TI/NI LogisticRegression с тем же подбором C; результаты — `retrained_ablations.csv`. Это **8 основных схем + 3 попарные абляции**.

C выбирается из [0.01, 0.1, 1] по validation macro-F1; веса late fusion — также по validation macro-F1. Нейросети: Adam, lr=0.001, weight decay=0.0001, batch=128, максимум 100 эпох, patience=10; checkpoint выбирается по **validation BCE**, минимум улучшения 1e-5. Веса обучаются только на train; validation не обновляет веса. Все сравнения используют threshold=0.5. После выбора нет train+validation refit.

Главный seed=42; нейросети дополнительно проверяются при 7 и 123. Оцениваются macro-F1, MCC, AUROC, Brier, accuracy и ECE (10 bins). Moving-block bootstrap: 1200 повторов, блоки 5/10/20 торговых сессий, все тикеры одной даты сохраняются вместе. Парные перестановки блоков проверяют одностороннюю гипотезу превосходства fusion; для пяти сравнений применяется Holm. Missing-modality стресс-тесты и переобученные абляции имеют разный смысл. Отсутствующий вход/полностью нулевой raw feature трактуется как missing, поэтому genuine all-zero неразличима.

Многокритериальный score — описательное test-сопоставление; модель выбирается по validation. Latency измеряется по готовым cached features, без чтения raw данных и извлечения признаков. `config.json` документирует фактические константы; **код его не читает**.

## Файлы

- `data/raw/kaggle_manifest.json` — источники, версии, provenance и SHA-256.
- `data/dataset.csv`, `article_alignment.csv`, `excluded_observations.csv`, `quality_summary.json`, `trading_calendar.csv` — объекты и аудит подготовки.
- `data/features.npz`, `numeric_windows.npz`, `encoders.joblib`, `images/` — признаки и обученные на train преобразования.
- `results/metrics.csv`, `predictions_*.csv`, `tuning.csv`, `models.joblib`, `run_summary.json` — основное обучение.
- `results/confidence_intervals.csv`, `delta_confidence_intervals.csv`, `significance.csv`, `retrained_ablations.csv`, `robustness.csv`, `seed_sensitivity.csv` — проверки выводов.
- `results/application_metrics.csv`, `error_examples.csv`, `reference_baselines.csv`, `resources.csv`, `multicriteria.csv` — применение, ошибки, ресурсы.
- `figures/` — рисунки; `report_lab2.docx` — итоговый отчёт.
- `research/literature.md` и `literature.json` — 15 первичных научных источников.

Связь с ВКР — оценка вклада найденных evidence и разных представлений в последующую задачу. Здесь retrieval-модель не экспериментируется; MIRACL искусственно в финансовый прогноз не добавляется.



Проверка готовых результатов: python src/verify.py. Сверка с методичкой: CHECKLIST.md. Полный фактический запуск и совпадение прогнозов: results/pipeline_status.json.


## Состав публикации на GitHub

В репозитории сохранены код, Word-отчёт, таблицы результатов, прогнозы, обученные компактные модели, подготовленные признаки, журнал исключений и итоговые графики. Исходные CSV/JSON Kaggle, архивы, установленный vendor и 3925 промежуточных свечных PNG не коммитятся. Они восстанавливаются командой `python run_all.py` после установки зависимостей. Научные источники и ограничения лицензий описаны в `research/` и manifest.
