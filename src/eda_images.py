"""EDA для датасета изображений (TrashNet и его обогащённых версий).

Структура данных: <root>/<класс>/<файл>.jpg — как в TrashNet.
Обогащённые версии (свои фото, сгенерированные изображения) надо
складывать в такую же структуру, тогда весь анализ переиспользуется.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan(root: str | Path, phash: bool = True) -> pd.DataFrame:
    """Обходит папку <root>/<класс>/<файл> и собирает метаданные и простые статистики."""
    root = Path(root)
    pairs = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        pairs += [(f, cls_dir.name) for f in sorted(cls_dir.iterdir()) if f.suffix.lower() in IMG_EXT]
    return scan_files(pairs, phash)


def scan_files(pairs, phash: bool = True) -> pd.DataFrame:
    """То же для явного списка (путь, метка); метка может быть None (ещё не размечено)."""
    rows = []
    for f, label in pairs:
        f = Path(f)
        row = {"path": str(f), "file": f.name, "label": label, "bytes": f.stat().st_size}
        try:
            with Image.open(f) as im:
                im.load()
                row.update(width=im.width, height=im.height, mode=im.mode)
                rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
                row.update(
                    mean_r=float(rgb[..., 0].mean()),
                    mean_g=float(rgb[..., 1].mean()),
                    mean_b=float(rgb[..., 2].mean()),
                    brightness=float(rgb.mean()),
                    contrast=float(rgb.std()),
                    # насыщенность как разброс между каналами — грубая, но быстрая оценка «цветности» кадра
                    colorfulness=float((rgb.max(-1) - rgb.min(-1)).mean()),
                )
                if phash:
                    row["phash"] = str(imagehash.phash(im))
            row["md5"] = hashlib.md5(f.read_bytes()).hexdigest()
            row["corrupted"] = False
        except Exception as e:  # noqa: BLE001 — нужно поймать любые битые файлы
            row["corrupted"] = True
            row["error"] = repr(e)
        rows.append(row)
    df = pd.DataFrame(rows)
    if "width" in df:
        df["aspect"] = df["width"] / df["height"]
    return df


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    vc = df["label"].value_counts().sort_index()
    return pd.DataFrame({"count": vc, "share": (vc / vc.sum()).round(4)})


def near_duplicates(df: pd.DataFrame, max_dist: int = 4) -> pd.DataFrame:
    """Пары почти одинаковых изображений по perceptual hash (расстояние Хэмминга ≤ max_dist)."""
    ok = df[~df["corrupted"]].reset_index(drop=True)
    hashes = np.array([imagehash.hex_to_hash(h).hash.flatten() for h in ok["phash"]])
    pairs = []
    for i in range(len(hashes)):
        d = (hashes[i + 1:] != hashes[i]).sum(axis=1)
        for j in np.nonzero(d <= max_dist)[0]:
            j = i + 1 + j
            pairs.append({
                "a": ok.at[i, "file"], "label_a": ok.at[i, "label"],
                "b": ok.at[j, "file"], "label_b": ok.at[j, "label"],
                "dist": int(d[j - i - 1]),
            })
    return pd.DataFrame(pairs, columns=["a", "label_a", "b", "label_b", "dist"])


def summary(df: pd.DataFrame, name: str, dups: pd.DataFrame | None = None) -> dict:
    ok = df[~df["corrupted"]]
    vc = ok["label"].value_counts()
    out = {
        "dataset": name,
        "n_images": int(len(df)),
        "n_corrupted": int(df["corrupted"].sum()),
        "n_classes": int(ok["label"].nunique()),
        "class_counts": {k: int(v) for k, v in vc.sort_index().items()},
        "imbalance_ratio_max_min": round(float(vc.max() / vc.min()), 2),
        "unique_resolutions": int(ok[["width", "height"]].drop_duplicates().shape[0]),
        "most_common_resolution": "x".join(map(str, ok[["width", "height"]].value_counts().index[0])),
        "brightness_mean": round(float(ok["brightness"].mean()), 4),
        "contrast_mean": round(float(ok["contrast"].mean()), 4),
        "colorfulness_mean": round(float(ok["colorfulness"].mean()), 4),
        "exact_duplicates_md5": int(ok["md5"].duplicated(keep=False).sum()),
    }
    if dups is not None:
        out["near_duplicate_pairs_phash"] = int(len(dups))
        out["near_duplicate_pairs_cross_class"] = int((dups["label_a"] != dups["label_b"]).sum())
    return out
