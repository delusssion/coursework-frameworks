"""Согласие разметчиков по выгрузке со страницы разметки.

Вход: annotations_all.csv (кнопка «Скачать всё (CSV)» на вкладке
«Согласие и арбитраж»), положить в data/annotation/.

Считает для каждого датасета и каждой пары разметчиков:
  * N — объекты, размеченные обоими;
  * P_o — доля совпадений;
  * P_e — ожидаемое случайное согласие по маргинальным частотам;
  * Cohen's kappa = (P_o - P_e) / (1 - P_e)  (сверяется со sklearn);
  * матрицу расхождений и список спорных объектов.
Результат: reports/annotation/agreement_<датасет>.json и .md

Запуск:  python -m src.agreement [путь к csv]
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "annotation" / "annotations_all.csv"
OUT = ROOT / "reports" / "annotation"

CATS = {
    "rusentitweet": ["positive", "negative", "neutral", "speech", "skip"],
    "trashnet": ["cardboard", "paper", "glass", "metal", "plastic", "trash", "exclude"],
}
BANDS = [(-1, 0, "хуже случайного"), (0, .2, "слабое"), (.2, .4, "удовлетворительное"),
         (.4, .6, "умеренное"), (.6, .8, "существенное"), (.8, 1.0001, "почти полное")]


def band(k: float) -> str:
    return next((name for lo, hi, name in BANDS if lo <= k < hi), "—")


def cohen_kappa(a: pd.Series, b: pd.Series, cats: list[str]) -> dict:
    n = len(a)
    po = float((a.values == b.values).mean())
    pa = a.value_counts(normalize=True).reindex(cats, fill_value=0)
    pb = b.value_counts(normalize=True).reindex(cats, fill_value=0)
    pe = float((pa * pb).sum())
    kappa = float("nan") if pe >= 1 else (po - pe) / (1 - pe)
    try:  # контроль своей реализации
        from sklearn.metrics import cohen_kappa_score
        sk = float(cohen_kappa_score(a, b, labels=cats))
        assert abs(sk - kappa) < 1e-9 or pe >= 1, (sk, kappa)
    except ImportError:
        pass
    return {"n": n, "p_o": round(po, 4), "p_e": round(pe, 4), "kappa": round(kappa, 4), "band": band(kappa)}


def analyse(df: pd.DataFrame, dataset: str) -> dict:
    cats = CATS[dataset]
    sub = df[df["dataset"] == dataset]
    anns = [c[len("label_"):] for c in sub.columns if c.startswith("label_")]
    anns = [x for x in anns if sub[f"label_{x}"].notna().any() and (sub[f"label_{x}"] != "").any()]
    result = {"dataset": dataset, "n_items": int(len(sub)), "annotators": {}, "pairs": []}
    for x in anns:
        lab = sub[f"label_{x}"].replace("", pd.NA).dropna()
        result["annotators"][x] = {"n_labeled": int(len(lab)),
                                   "distribution": {k: int(v) for k, v in lab.value_counts().items()},
                                   "unsure": int(pd.to_numeric(sub.get(f"unsure_{x}"), errors="coerce").fillna(0).sum())}
    for x, y in itertools.combinations(anns, 2):
        both = sub[[f"label_{x}", f"label_{y}", "item_id", "text", "asset_id", "calibration"]].replace("", pd.NA)
        both = both.dropna(subset=[f"label_{x}", f"label_{y}"])
        if both.empty:
            continue
        a, b = both[f"label_{x}"], both[f"label_{y}"]
        stats = cohen_kappa(a, b, cats)
        cal = both["calibration"].astype(str) == "1"
        if cal.any() and (~cal).any():
            stats["kappa_calibration"] = cohen_kappa(a[cal], b[cal], cats)["kappa"]
            stats["kappa_main"] = cohen_kappa(a[~cal], b[~cal], cats)["kappa"]
        cm = pd.crosstab(a, b).reindex(index=cats, columns=cats, fill_value=0)
        used = [c for c in cats if cm.loc[c].sum() or cm[c].sum()]
        dis = both[a != b]
        result["pairs"].append({
            "a": x, "b": y, **stats,
            "confusion": cm.loc[used, used].to_dict(orient="index"),
            "disagreements": [
                {"item_id": r.item_id, "content": r.text if isinstance(r.text, str) else r.asset_id,
                 "a": r[f"label_{x}"], "b": r[f"label_{y}"]}
                for _, r in dis.iterrows()
            ],
        })
    final = sub["final"].replace("", pd.NA).dropna() if "final" in sub else pd.Series(dtype=str)
    result["final_labels"] = {k: int(v) for k, v in final.value_counts().items()}
    return result


def to_md(r: dict) -> str:
    lines = [f"# Согласие разметчиков — {r['dataset']}", "", f"Объектов в выгрузке: {r['n_items']}", ""]
    for name, a in r["annotators"].items():
        lines.append(f"- **{name}**: размечено {a['n_labeled']}, «сомневаюсь» {a['unsure']}, распределение {a['distribution']}")
    for p in r["pairs"]:
        lines += ["", f"## {p['a']} × {p['b']}", "",
                  "| N | P_o | P_e | κ | интерпретация |", "|---|---|---|---|---|",
                  f"| {p['n']} | {p['p_o']:.3f} | {p['p_e']:.3f} | {p['kappa']:.3f} | {p['band']} |"]
        if "kappa_calibration" in p:
            lines.append(f"\nκ на калибровке: {p['kappa_calibration']:.3f}; на основной разметке: {p['kappa_main']:.3f}")
        cm = pd.DataFrame(p["confusion"]).T
        lines += ["", "Матрица расхождений (строки — " + p["a"] + ", столбцы — " + p["b"] + "):", "", cm.to_markdown()]
        lines += ["", f"Расхождений: {len(p['disagreements'])}"]
    if r["final_labels"]:
        lines += ["", f"Итоговые метки после арбитража: {r['final_labels']}"]
    return "\n".join(lines) + "\n"


def main(path: Path) -> None:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in CATS:
        if not (df["dataset"] == ds).any():
            continue
        r = analyse(df, ds)
        (OUT / f"agreement_{ds}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        (OUT / f"agreement_{ds}.md").write_text(to_md(r), encoding="utf-8")
        for p in r["pairs"]:
            print(f"{ds}: {p['a']} × {p['b']}  N={p['n']}  P_o={p['p_o']:.3f}  kappa={p['kappa']:.3f} ({p['band']})")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV)
