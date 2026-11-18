"""Генерирует notebooks/01_eda_before_enrichment.ipynb (затем он исполняется nbconvert)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# EDA исходных датасетов (до обогащения)

**Курсовая работа:** «Сравнительный анализ влияния ручного и искусственного обогащения данных на качество и устойчивость обучения моделей машинного обучения».

**Этап 1 (21.10.26):** выбор датасетов, журнал работы, предварительный список источников реальных данных, начало анализа исходных датасетов.

| | Датасет 1 — текст | Датасет 2 — изображения |
|---|---|---|
| Название | RuSentiTweet | TrashNet |
| Задача | классификация тональности, 5 классов | классификация изображений, 6 классов |
| Объём | 13 392 твита | 2 527 фото |
| Источник | github.com/sismetanin/rusentitweet | github.com/garythung/trashnet |
| Лицензия | открытый датасет к статье в PeerJ CS (2022), цитирование обязательно | MIT |

Все метрики считаются функциями из `src/`, сводки сохраняются в `reports/eda_before/*.json` — после обогащения тот же код запускается на новых данных, и сводки сравниваются.
""")

code(r"""
import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from src import download, eda_text as T, eda_images as I

OUT = ROOT / "reports" / "eda_before"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.max_colwidth", 120)

def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
""")

code(r"""
# Скачивает данные в data/raw, если их ещё нет (пропускает уже скачанное)
download.download_rusentitweet()
download.download_trashnet();
""")

# ---------------- TEXT ----------------
md(r"""
## 1. RuSentiTweet

Твиты на русском из архива Twitter Stream Grab, размечены краудсорсингом по инструкции RuSentiment (фактически — готовый codebook). Классы:

- `positive` / `negative` / `neutral` — тональность автора;
- `speech` — речевой акт: благодарность, поздравление, приветствие;
- `skip` — нерелевантное: шум, другой язык, непонятный текст.

По данным авторов (ноутбук «Labels Aggregation» в репозитории), согласие разметчиков — Krippendorff $\alpha \approx 0{,}505$ для номинальной шкалы, то есть умеренное. Это важно для курсовой: исходные метки сами по себе зашумлены, и шум разметки нужно учитывать при оценке эффекта обогащения.
""")

code(r"""
full = pd.read_csv(ROOT / "data/raw/rusentitweet/rusentitweet_full.csv", index_col=0)
train = pd.read_csv(ROOT / "data/raw/rusentitweet/rusentitweet_train.csv")
test = pd.read_csv(ROOT / "data/raw/rusentitweet/rusentitweet_test.csv")
print("full:", full.shape, " train:", train.shape, " test:", test.shape)
print("Пересечение id train/test:", len(set(train.id) & set(test.id)))
print("Пропуски:\n", full.isna().sum().to_string())
full.sample(8, random_state=0)
""")

md("### 1.1. Распределение классов")
code(r"""
dist = T.class_distribution(full)
split = pd.concat({
    "train": train["label"].value_counts(normalize=True),
    "test": test["label"].value_counts(normalize=True),
}, axis=1).reindex(dist.index).round(4)
display(dist.join(split))
print("Дисбаланс max/min:", round(T.imbalance_ratio(full), 2))

fig, ax = plt.subplots(figsize=(7, 3.5))
sns.barplot(x=dist.index, y=dist["count"], ax=ax, color="#4C72B0")
for i, v in enumerate(dist["count"]):
    ax.text(i, v, f"{v}\n({dist['share'].iloc[i]:.1%})", ha="center", va="bottom", fontsize=9)
ax.set(title="RuSentiTweet: число объектов по классам", xlabel="", ylabel="объектов")
ax.set_ylim(0, dist["count"].max() * 1.2)
save(fig, "text_class_distribution")
""")

md(r"""
Классы сильно несбалансированы: `neutral` почти 40 %, `speech` — меньше 4 % (дисбаланс $\approx 10{,}8$ раза). Разбиение train/test стратифицировано — доли совпадают, утечки по `id` нет. Значит, главная цель обогащения для текстов — **малые классы `speech` и `positive`**, и метрикой качества дальше должна быть macro-F1, а не accuracy.
""")

md("### 1.2. Длина текстов и особенности твитов")
code(r"""
F = T.add_features(full)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
sns.boxplot(data=F, x="label", y="n_words", order=dist.index, ax=axes[0], showfliers=False, color="#8DA0CB")
axes[0].set(title="Число слов (без ссылок и @)", xlabel="", ylabel="слов")
sns.histplot(F["n_chars"], bins=60, ax=axes[1], color="#66C2A5")
axes[1].set(title="Длина в символах", xlabel="символов", ylabel="твитов")
save(fig, "text_lengths")

feat = F.groupby("label")[["n_chars", "n_words"]].median().join(
    F.groupby("label")[["has_url", "has_mention", "has_hashtag"]].mean().round(3)
).join((F.groupby("label")["n_emoji"].apply(lambda s: (s > 0).mean())).rename("has_emoji").round(3)).reindex(dist.index)
feat
""")

md(r"""
Тексты короткие — медиана 6 слов. Около 60 % твитов содержат упоминания `@user`, около 20 % — ссылки: это шум, который нужно чистить или заменять токенами. Эмодзи встречаются в ~18 % твитов и заметно связаны с тональностью — их стоит оставить как признак. Распределение обрывается около 140 символов — это старый лимит Twitter, твиты собраны до его повышения.
""")

md("### 1.3. Язык и класс `skip`")
code(r"""
fig, ax = plt.subplots(figsize=(7, 3.3))
sns.boxplot(data=F, x="label", y="cyr_share", order=dist.index, ax=ax, color="#FC8D62")
ax.set(title="Доля кириллицы среди букв", xlabel="", ylabel="доля")
save(fig, "text_cyrillic_share")
print("Твитов с долей кириллицы < 0.5:", int((F["cyr_share"] < 0.5).sum()))
F[F["label"] == "skip"].sample(8, random_state=1)[["text", "label"]]
""")

md(r"""
В `skip` попадают твиты на других языках, прежде всего монгольском (кириллица, но слова «байна», «гэж», «дээ»), и бессодержательные ответы. Для обогащения это значит: реальные данные, собранные вручную, придётся фильтровать по языку, иначе они раздуют `skip`.
""")

md("### 1.4. Частотные слова по классам")
code(r"""
tops = {lab: [w for w, _ in T.top_tokens(full, lab, 12)] for lab in dist.index}
pd.DataFrame(tops)
""")

md(r"""
Лексика классов ожидаемая: `speech` — «спасибо», «поздравляю», «доброе утро»; `negative` — в основном обсценная лексика. Это значит, что модель легко научится на ключевых словах, а обогащение должно добавлять примеры **без** очевидных маркеров (негатив без мата, позитив без «люблю»), чтобы проверять устойчивость.
""")

md("### 1.5. Дубликаты и противоречивые метки")
code(r"""
dups = T.duplicates_report(full)
print(json.dumps(dups, ensure_ascii=False, indent=1))

norm = full["text"].map(T.clean_for_tokens).str.replace(r"\s+", " ", regex=True).str.strip()
g = full.assign(norm=norm).groupby("norm")["label"].agg(["nunique", "count", lambda s: ", ".join(sorted(set(s)))])
g.columns = ["n_labels", "n_rows", "labels"]
g[g["n_labels"] > 1].sort_values("n_rows", ascending=False).head(10)
""")

md(r"""
Точных дублей нет, но после удаления ссылок и упоминаний 328 строк совпадают, и в 38 группах одинаковый текст имеет **разные метки**. Это прямое подтверждение шума разметки ($\alpha \approx 0{,}5$). При ручной доразметке эти случаи — готовый материал для раздела codebook «спорные примеры».
""")

code(r"""
text_summary = T.summary(full, "rusentitweet_full")
(OUT / "text_summary.json").write_text(json.dumps(text_summary, ensure_ascii=False, indent=2))
text_summary
""")

# ---------------- IMAGES ----------------
md(r"""
## 2. TrashNet

Фотографии бытовых отходов на белом фоне, сделанные авторами вручную (Stanford, 2017), 6 классов: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`. В репозитории — версия, уменьшенная до 512×384, и списки разбиения train/val/test.

Датасет хорошо подходит под тему курсовой: съёмка однородная (один фон, одно освещение), поэтому можно прямо проверить, улучшает ли обогащение **устойчивость к смене домена** — реальными фото в других условиях (ручное обогащение) или аугментациями и генерацией (искусственное).
""")

code(r"""
IMG_ROOT = ROOT / "data/raw/trashnet/dataset-resized"
img = I.scan(IMG_ROOT)
print(img.shape, "| битых файлов:", int(img["corrupted"].sum()))
img.head()
""")

md("### 2.1. Распределение классов и разбиение")
code(r"""
idist = I.class_distribution(img)
splits = {}
for part in ["train", "val", "test"]:
    lines = (ROOT / f"data/raw/trashnet/one-indexed-files-notrash_{part}.txt").read_text().split()
    names = lines[0::2]
    splits[part] = pd.Series(names).str.replace(r"\d+\.jpg$", "", regex=True).value_counts()
sp = pd.DataFrame(splits).reindex(idist.index).fillna(0).astype(int)
display(idist.join(sp))
print("Дисбаланс max/min:", round(idist["count"].max() / idist["count"].min(), 2))

fig, ax = plt.subplots(figsize=(7, 3.5))
sns.barplot(x=idist.index, y=idist["count"], ax=ax, color="#55A868")
for i, v in enumerate(idist["count"]):
    ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
ax.set(title="TrashNet: число изображений по классам", xlabel="", ylabel="изображений")
save(fig, "img_class_distribution")
""")

md(r"""
Дисбаланс умеренный ($\approx 4{,}3$ раза), самый маленький класс — `trash` (137 фото, 5,4 %). Авторское разбиение примерно 70/13/17 и стратифицировано; его можно взять как есть для воспроизводимости, но дубликаты (раздел 2.4) нужно убрать до разбиения, иначе одно и то же фото может попасть и в train, и в test. Дополнительно понадобится отдельный тест из реальных фото «в другом домене».
""")

md("### 2.2. Примеры")
code(r"""
rng = np.random.default_rng(0)
classes = list(idist.index)
fig, axes = plt.subplots(len(classes), 5, figsize=(11, 2.1 * len(classes)))
for r, c in enumerate(classes):
    paths = img[img.label == c]["path"].to_numpy()
    for k, p in enumerate(rng.choice(paths, 5, replace=False)):
        axes[r, k].imshow(Image.open(p)); axes[r, k].axis("off")
    axes[r, 0].set_title(c, loc="left", fontsize=10)
plt.tight_layout()
save(fig, "img_examples")
""")

md("### 2.3. Размеры, яркость, цвет")
code(r"""
print("Разрешения:", img[["width", "height"]].value_counts().to_dict())
print("Режимы:", img["mode"].value_counts().to_dict())
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, col, title in zip(axes, ["brightness", "contrast", "colorfulness"],
                          ["Средняя яркость", "Контраст (std)", "Цветность"]):
    sns.boxplot(data=img, x="label", y=col, ax=ax, color="#CCB974", showfliers=False)
    ax.set(title=title, xlabel="", ylabel=""); ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
save(fig, "img_color_stats")
img.groupby("label")[["brightness", "contrast", "colorfulness", "bytes"]].median().round(3)
""")

md(r"""
Все изображения одного разрешения 512×384 и в RGB — предобработка размеров не нужна. Яркость высокая и почти одинаковая по классам (белый фон), цветность низкая. Это и есть главный риск: модель может опираться на условия съёмки, а не на материал. Реальные фото с цветным фоном и другим освещением должны дать заметное падение качества — его и будем «лечить» обогащением.
""")

md("### 2.4. Дубликаты, в том числе между классами")
code(r"""
nd = I.near_duplicates(img, max_dist=4)
print("Точных дублей (md5), строк:", int(img["md5"].duplicated(keep=False).sum()))
print("Пар почти одинаковых (pHash, расстояние ≤ 4):", len(nd))
cross = nd[nd.label_a != nd.label_b]
print("Из них в разных классах:", len(cross))
cross
""")

code(r"""
show = cross.head(5)
if len(show):
    fig, axes = plt.subplots(len(show), 2, figsize=(6, 2.3 * len(show)))
    axes = np.atleast_2d(axes)
    for i, (_, r) in enumerate(show.iterrows()):
        for j, (f, l) in enumerate([(r.a, r.label_a), (r.b, r.label_b)]):
            axes[i, j].imshow(Image.open(IMG_ROOT / l / f)); axes[i, j].axis("off")
            axes[i, j].set_title(f"{l}/{f}  (d={r.dist})", fontsize=9)
    plt.tight_layout()
    save(fig, "img_cross_class_duplicates")
""")

md(r"""
Пары с расстоянием 0 — это **одно и то же фото, лежащее в двух классах**: `glass115` = `metal91`, `glass176` = `plastic152`, `glass389` = `plastic332`. Это ошибки разметки исходного датасета: по фото видно, что во всех трёх случаях это стекло или пластиковая бутылка, а не металл. Пары с расстоянием 2–4 на белом фоне — частые ложные срабатывания pHash (например, `metal154` и `plastic10` — разные предметы), поэтому кандидатов нужно смотреть глазами, а не удалять автоматически. Решение по ошибочным файлам (удалить или переразметить по codebook) фиксируем в журнале.
""")

code(r"""
img_summary = I.summary(img, "trashnet_resized", nd)
(OUT / "image_summary.json").write_text(json.dumps(img_summary, ensure_ascii=False, indent=2))
nd.to_csv(OUT / "image_near_duplicates.csv", index=False)
img_summary
""")

md(r"""
## 3. Выводы этапа 1

**RuSentiTweet**
- 13 392 коротких твита (медиана 6 слов), 5 классов, сильный дисбаланс: `speech` 3,7 %, `neutral` 39,9 %.
- Исходная разметка шумная: $\alpha \approx 0{,}505$, 38 групп одинаковых текстов с разными метками.
- Много шума Twitter: упоминания (60 %), ссылки (21 %), чужие языки в `skip`.
- Цели обогащения: классы `speech` и `positive`, примеры без явных ключевых слов.

**TrashNet**
- 2 527 фото, 6 классов, один размер и одинаковые условия съёмки (белый фон).
- Класс `trash` маленький (137 фото, 5,4 %).
- Есть дубликаты, в том числе 3 фото, продублированных в разных классах, — ошибки разметки.
- Цели обогащения: класс `trash`, фото в других условиях (фон, освещение, ракурс) для проверки устойчивости.

**Дальше:** базовые модели (TF-IDF + логистическая регрессия; дообученная CNN), codebook для ручной доразметки, сбор реальных данных по списку источников, искусственное обогащение, сравнение по macro-F1 и на отдельном «доменном» тесте.
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
Path(__file__).with_name("01_eda_before_enrichment.ipynb").write_text(nbf.writes(nb), encoding="utf-8")
print("ok")
