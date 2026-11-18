# Ручное и искусственное обогащение данных: влияние на качество и устойчивость моделей

Курсовая работа по курсу «Фреймворки Python для анализа данных».

**Тема:** Сравнительный анализ влияния ручного и искусственного обогащения данных на качество и устойчивость обучения моделей машинного обучения.

**Исполнители:** Шалыгин Ярослав, Соколов Александр · **Журнал работы:** [JOURNAL.md](JOURNAL.md)

## Датасеты

| Модальность | Датасет | Задача | Объём | Лицензия |
|---|---|---|---|---|
| Текст | [RuSentiTweet](https://github.com/sismetanin/rusentitweet) | тональность, 5 классов | 13 392 твита | открытый, цитирование статьи |
| Изображения | [TrashNet](https://github.com/garythung/trashnet) | материал отходов, 6 классов | 2 527 фото | MIT |

Подробности и обоснование выбора — [docs/datasets.md](docs/datasets.md).

## Структура

```
├── JOURNAL.md                         журнал работы
├── data/raw/                          исходные данные (не в git, скачиваются скриптом)
├── docs/
│   ├── datasets.md                    карточки датасетов, проверка на повторы
│   ├── sources_enrichment.md          источники реальных данных для обогащения
│   ├── codebook_text.md               codebook разметки текстов
│   └── codebook_images.md             codebook разметки фото
├── notebooks/
│   ├── 01_eda_before_enrichment.ipynb EDA исходных датасетов
│   ├── 02_eda_real_enrichment.ipynb   анализ до и после ручного обогащения
│   └── build_*.py                     генераторы ноутбуков
├── reports/eda_before/                графики и JSON-сводки EDA «до обогащения»
├── data/enriched/real/               ручное обогащение: 70 текстов, 63 фото (+ служебные файлы)
├── data/annotation/                   метки разметчиков A и B, арбитраж, сводная таблица
├── reports/annotation/                согласие разметчиков (κ, матрицы расхождений)
├── reports/eda_after/                 графики анализа после обогащения
└── src/
    ├── collect_real.py                отбор реальных данных для обогащения
    ├── annotation_sheets.py           (опционально) листы разметки Excel для людей
    ├── annotation_merge.py            сборка разметки, κ, арбитраж, итоговый датасет
    ├── agreement.py                   согласие разметчиков (P_o, Cohen's κ, матрица)
    ├── download.py                    загрузка датасетов
    ├── eda_text.py                    EDA текстов
    └── eda_images.py                  EDA изображений
```

## Запуск

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.download                 # ~45 МБ, в data/raw
jupyter lab notebooks/01_eda_before_enrichment.ipynb
```

Ноутбук сам скачивает данные, если их нет. Анализ «после обогащения» делается теми же функциями из `src/` на новой папке данных — сводки в `reports/` сравниваются.

## Используемые библиотеки и фреймворки

| Библиотека | Версия | Для чего |
|---|---|---|
| Python | 3.11+ | язык |
| pandas | 3.0.5 | табличные данные |
| NumPy | 2.x | вычисления |
| Matplotlib | 3.11.2 | графики |
| seaborn | 0.13.2 | статистические графики |
| Pillow | 12.3.0 | чтение изображений |
| ImageHash | 4.3.2 | поиск дубликатов изображений (pHash) |
| scikit-learn | 1.x | проверка Cohen's κ |
| tabulate | 0.9 | таблицы в отчётах Markdown |
| openpyxl | 3.x | листы разметки Excel |
| JupyterLab, nbconvert | — | ноутбуки |

Будут добавлены на следующих этапах: PyTorch + torchvision (CNN), albumentations (аугментации), Label Studio (разметка), Telethon / VK API (сбор данных).

## Ручное обогащение и разметка

```bash
python -m src.collect_real        # отбор реальных данных -> data/enriched/real/
# метки разметчиков: data/annotation/labels_A_shalygin.csv, labels_B_sokolov.csv; арбитраж: arbitration.csv
python -m src.annotation_merge    # κ, отчёт в reports/annotation/, итоговые *_labeled.csv
```

Источники: тексты — [skoltech-nlp/inappropriate-sensitive-topics](https://github.com/skoltech-nlp/inappropriate-sensitive-topics) (CC BY-NC-SA 4.0), фото — [TrashBox](https://github.com/nikhilvenkatkumsetty/TrashBox) (при использовании требуется ссылка на статью авторов).

Разметка: разметчик A — Шалыгин Ярослав (`labels_A_shalygin.csv`), разметчик B — Соколов Александр (`labels_B_sokolov.csv`), по codebook v1.1; расхождения разобраны вместе (`arbitration.csv`). Подробности — в журнале.

## Использование генеративных моделей

Протокол — [docs/llm_protocol.md](docs/llm_protocol.md).
