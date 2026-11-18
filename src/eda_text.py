"""EDA для текстового датасета (RuSentiTweet и его обогащённых версий).

Все функции принимают DataFrame с колонками ``text`` и ``label`` и
возвращают числа/таблицы, поэтому один и тот же код запускается
на исходных данных и на данных после обогащения — так получается
сравнение «до / после», которое требуется в курсовой.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

LABEL_ORDER = ["positive", "neutral", "negative", "speech", "skip"]

RE_URL = re.compile(r"https?://\S+")
RE_MENTION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#\w+")
RE_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)
RE_WORD = re.compile(r"[а-яёa-z]+", re.IGNORECASE)
RE_CYR = re.compile(r"[а-яё]", re.IGNORECASE)
RE_LAT = re.compile(r"[a-z]", re.IGNORECASE)

# Небольшой список стоп-слов, чтобы не тянуть nltk ради одного шага
STOPWORDS = set(
    """и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
    только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если
    уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей
    может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз
    тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом
    один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец
    два об другой хоть после над больше тот через эти нас про всего них какая много
    разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой
    им более всегда конечно всю между это так просто rt""".split()
)


def clean_for_tokens(text: str) -> str:
    text = RE_URL.sub(" ", text)
    text = RE_MENTION.sub(" ", text)
    return text.lower()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет к датафрейму признаки текста, используемые в EDA."""
    out = df.copy()
    t = out["text"].fillna("").astype(str)
    out["n_chars"] = t.str.len()
    out["n_words"] = t.map(lambda s: len(RE_WORD.findall(clean_for_tokens(s))))
    out["has_url"] = t.str.contains(RE_URL)
    out["has_mention"] = t.str.contains(RE_MENTION)
    out["has_hashtag"] = t.str.contains(RE_HASHTAG)
    out["n_emoji"] = t.map(lambda s: len(RE_EMOJI.findall(s)))
    cyr = t.map(lambda s: len(RE_CYR.findall(s)))
    lat = t.map(lambda s: len(RE_LAT.findall(clean_for_tokens(s))))
    out["cyr_share"] = cyr / (cyr + lat).replace(0, np.nan)
    # «пустой» после удаления ссылок и упоминаний текст — частый источник метки skip
    out["is_empty_after_clean"] = out["n_words"] == 0
    return out


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    vc = df["label"].value_counts()
    order = [l for l in LABEL_ORDER if l in vc.index] + [l for l in vc.index if l not in LABEL_ORDER]
    vc = vc.reindex(order)
    return pd.DataFrame({"count": vc, "share": (vc / vc.sum()).round(4)})


def imbalance_ratio(df: pd.DataFrame) -> float:
    vc = df["label"].value_counts()
    return float(vc.max() / vc.min())


def duplicates_report(df: pd.DataFrame) -> dict:
    norm = df["text"].fillna("").map(clean_for_tokens).str.replace(r"\s+", " ", regex=True).str.strip()
    exact = df["text"].duplicated(keep=False)
    near = norm.duplicated(keep=False) & (norm != "")
    # одинаковый нормализованный текст, но разные метки — прямой индикатор шума разметки
    grp = pd.DataFrame({"norm": norm, "label": df["label"]})[near].groupby("norm")["label"].nunique()
    return {
        "exact_dup_rows": int(exact.sum()),
        "normalized_dup_rows": int(near.sum()),
        "dup_groups_with_conflicting_labels": int((grp > 1).sum()),
    }


def top_tokens(df: pd.DataFrame, label: str | None = None, k: int = 15) -> list[tuple[str, int]]:
    sub = df if label is None else df[df["label"] == label]
    cnt: Counter = Counter()
    for s in sub["text"].fillna(""):
        cnt.update(w for w in RE_WORD.findall(clean_for_tokens(s)) if w not in STOPWORDS and len(w) > 2)
    return cnt.most_common(k)


def summary(df: pd.DataFrame, name: str) -> dict:
    """Короткая сводка — сохраняется в JSON как точка «до обогащения»."""
    f = add_features(df)
    return {
        "dataset": name,
        "n_rows": int(len(f)),
        "n_classes": int(f["label"].nunique()),
        "class_counts": {k: int(v) for k, v in f["label"].value_counts().items()},
        "imbalance_ratio_max_min": round(imbalance_ratio(f), 2),
        "chars_median": float(f["n_chars"].median()),
        "words_median": float(f["n_words"].median()),
        "share_with_url": round(float(f["has_url"].mean()), 4),
        "share_with_mention": round(float(f["has_mention"].mean()), 4),
        "share_with_emoji": round(float((f["n_emoji"] > 0).mean()), 4),
        "share_empty_after_clean": round(float(f["is_empty_after_clean"].mean()), 4),
        "vocab_size": int(len({w for s in f["text"].fillna("") for w in RE_WORD.findall(clean_for_tokens(s))})),
        **duplicates_report(f),
    }
