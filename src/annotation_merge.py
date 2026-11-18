"""Сборка разметки двух разметчиков, арбитраж и итоговый датасет.

Метки разметчиков: data/annotation/labels_<разметчик>.csv
(item_id, dataset, label, unsure, comment) или заполненные листы
data/annotation/разметка_<разметчик>.xlsx.
Арбитраж: data/annotation/arbitration.csv (item_id, final, rationale);
если его нет — используется колонка «Итог» в арбитраж.xlsx.

Шаги:
1. Положить метки двух разметчиков в data/annotation/.
2. python -m src.annotation_merge
   -> data/annotation/annotations_all.csv   (все метки рядом; вход для src.agreement)
   -> reports/annotation/agreement_*.md/json (P_o, Cohen's κ, матрица)
   -> data/annotation/арбитраж.xlsx         (только расхождения; колонка «Итог» пустая)
3. Вдвоём заполнить «Итог» в арбитраж.xlsx по codebook, снова запустить скрипт
   -> data/enriched/real/texts_labeled.csv, images_labeled.csv (итоговые метки, без exclude)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from src import agreement
from src.annotation_sheets import CAL_N, IMG_LABELS, REAL, TEXT_LABELS

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data" / "annotation"
ARB = ANN / "арбитраж.xlsx"
SHEETS = {"Тексты": ("rusentitweet", {l[0] for l in TEXT_LABELS}),
          "Фото": ("trashnet", {l[0] for l in IMG_LABELS})}


def read_sheet(path: Path) -> pd.DataFrame:
    wb = load_workbook(path, data_only=True)
    who = path.stem.split("_", 1)[1]
    rows = []
    for sheet, (ds, valid) in SHEETS.items():
        ws = wb[sheet]
        for r in ws.iter_rows(min_row=4, values_only=True):
            if not r or r[1] is None:
                continue
            lab = (str(r[3]).strip().lower() if r[3] is not None else "")
            if lab and lab not in valid:
                raise ValueError(f"{path.name}, лист {sheet}, {r[1]}: неизвестная метка «{r[3]}»")
            rows.append({"item_id": r[1], "dataset": ds, "annotator": who, "label": lab,
                         "unsure": 1 if str(r[4] or "").strip().lower() in {"да", "1", "yes"} else 0,
                         "comment": r[5] or ""})
    return pd.DataFrame(rows)


def read_csv_labels(path: Path) -> pd.DataFrame:
    who = path.stem.split("_", 1)[1]
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    for ds, valid in [(v[0], v[1]) for v in SHEETS.values()]:
        bad = d[(d["dataset"] == ds) & ~d["label"].isin(valid)]
        if len(bad):
            raise ValueError(f"{path.name}: неизвестные метки {bad[['item_id', 'label']].values.tolist()}")
    d["unsure"] = d["unsure"].map(lambda x: 1 if str(x).strip() in {"1", "да", "true"} else 0)
    return d.assign(annotator=who)[["item_id", "dataset", "annotator", "label", "unsure", "comment"]]


def items() -> pd.DataFrame:
    t = pd.read_csv(REAL / "texts.csv")
    t = pd.DataFrame({"item_id": t["id"], "dataset": "rusentitweet", "seq": range(1, len(t) + 1),
                      "text": t["text"], "asset_id": "", "source": t["source"]})
    m = pd.read_csv(REAL / "images_manifest.csv").sort_values("order")
    i = pd.DataFrame({"item_id": m["id"], "dataset": "trashnet", "seq": m["order"].values,
                      "text": "", "asset_id": m["file"], "source": m["source"]})
    df = pd.concat([t, i], ignore_index=True)
    df["calibration"] = (df["seq"] <= CAL_N).astype(int)
    return df


def read_arbitration() -> dict:
    csv_path = ANN / "arbitration.csv"
    if csv_path.exists():
        a = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        return {r.item_id: r.final.strip().lower() for r in a.itertuples() if r.final.strip()}
    if not ARB.exists():
        return {}
    ws = load_workbook(ARB, data_only=True).active
    out = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0] and r[-1]:
            out[r[0]] = str(r[-1]).strip().lower()
    return out


def write_arbitration(wide: pd.DataFrame, names: list[str], done: dict) -> int:
    a, b = names[:2]
    dis = wide[(wide[f"label_{a}"] != "") & (wide[f"label_{b}"] != "") & (wide[f"label_{a}"] != wide[f"label_{b}"])]
    wb = Workbook()
    ws = wb.active
    ws.title = "Арбитраж"
    head = ["item_id", "Датасет", "Объект", f"Метка: {a}", f"Комментарий: {a}", f"Метка: {b}", f"Комментарий: {b}", "Итог"]
    ws.append(head)
    for c in ws[1]:
        c.font = Font(name="Arial", bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="2F6B4F")
    for k, (_, d) in enumerate(dis.iterrows(), 2):
        ws.append([d["item_id"], d["dataset"], d["text"] if d["dataset"] == "rusentitweet" else "",
                   d[f"label_{a}"], d[f"comment_{a}"], d[f"label_{b}"], d[f"comment_{b}"], done.get(d["item_id"])])
        if d["dataset"] == "trashnet":
            img = XLImage(str(REAL / d["asset_id"]))
            img.width, img.height = img.width * 0.35, img.height * 0.35
            ws.add_image(img, f"C{k}")
            ws.row_dimensions[k].height = 140
        for c in ws[k]:
            c.font = Font(name="Arial", size=10)
            c.alignment = Alignment(wrap_text=True, vertical="top")
        ws[f"H{k}"].fill = PatternFill("solid", fgColor="FFF6CC")
    labels = [l[0] for l in TEXT_LABELS + IMG_LABELS]
    dv = DataValidation(type="list", formula1='"' + ",".join(labels) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"H2:H{max(2, len(dis) + 1)}")
    for col, w in zip("ABCDEFGH", [8, 13, 60, 12, 24, 12, 24, 12]):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A2"
    wb.save(ARB)
    return len(dis)


def main() -> None:
    parts = [read_csv_labels(p) for p in sorted(ANN.glob("labels_*.csv"))]
    parts += [read_sheet(p) for p in sorted(ANN.glob("разметка_*.xlsx"))]
    if len(parts) < 2:
        raise SystemExit("Нужны метки двух разметчиков в data/annotation/ (labels_*.csv или разметка_*.xlsx).")
    long = pd.concat(parts, ignore_index=True)
    names = sorted(long["annotator"].unique())
    wide = items()
    for n in names:
        part = long[long["annotator"] == n].set_index(["item_id", "dataset"])[["label", "unsure", "comment"]]
        part.columns = [f"{c}_{n}" for c in part.columns]
        wide = wide.join(part, on=["item_id", "dataset"])
    wide = wide.fillna("")
    a, b = names[:2]
    done = read_arbitration()
    agree = (wide[f"label_{a}"] != "") & (wide[f"label_{a}"] == wide[f"label_{b}"])
    wide["final"] = ""
    wide.loc[agree, "final"] = wide.loc[agree, f"label_{a}"]
    for iid, lab in done.items():
        wide.loc[wide["item_id"] == iid, "final"] = lab
    ANN.mkdir(parents=True, exist_ok=True)
    wide.to_csv(ANN / "annotations_all.csv", index=False)
    agreement.main(ANN / "annotations_all.csv")
    n_dis = write_arbitration(wide, names, done)
    left = int(((wide[f"label_{a}"] != "") & (wide[f"label_{b}"] != "") & (wide["final"] == "")).sum())
    print(f"Расхождений: {n_dis}, не решено арбитражем: {left}  ->  {ARB.relative_to(ROOT)}")
    if left == 0:
        t = wide[(wide["dataset"] == "rusentitweet") & (wide["final"] != "")][["item_id", "text", "final", "source"]]
        t.rename(columns={"final": "label"}).to_csv(REAL / "texts_labeled.csv", index=False)
        im = wide[(wide["dataset"] == "trashnet") & (wide["final"] != "") & (wide["final"] != "exclude")]
        im[["item_id", "asset_id", "final", "source"]].rename(columns={"final": "label", "asset_id": "file"}) \
            .to_csv(REAL / "images_labeled.csv", index=False)
        print(f"Итог: текстов {len(t)}, фото {len(im)} (исключено {int((wide['final'] == 'exclude').sum())})")


if __name__ == "__main__":
    main()
