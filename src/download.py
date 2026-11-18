"""Загрузка исходных датасетов в data/raw.

Оба датасета лежат в открытых GitHub-репозиториях авторов, поэтому
скачиваются без ключей и регистрации:

* RuSentiTweet  — https://github.com/sismetanin/rusentitweet
* TrashNet      — https://github.com/garythung/trashnet (лицензия MIT)

Запуск:  python -m src.download
"""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

RUSENTI_BASE = "https://raw.githubusercontent.com/sismetanin/rusentitweet/main/"
RUSENTI_FILES = ["rusentitweet_full.csv", "rusentitweet_train.csv", "rusentitweet_test.csv"]

TRASHNET_BASE = "https://raw.githubusercontent.com/garythung/trashnet/master/data/"
TRASHNET_FILES = [
    "dataset-resized.zip",
    "one-indexed-files-notrash_train.txt",
    "one-indexed-files-notrash_val.txt",
    "one-indexed-files-notrash_test.txt",
]


def _fetch(url: str, dst: Path) -> None:
    if dst.exists():
        print(f"[skip] {dst.relative_to(ROOT)} уже есть")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {url}")
    urllib.request.urlretrieve(url, dst)


def download_rusentitweet() -> Path:
    out = RAW / "rusentitweet"
    for name in RUSENTI_FILES:
        _fetch(RUSENTI_BASE + name, out / name)
    return out


def download_trashnet() -> Path:
    out = RAW / "trashnet"
    for name in TRASHNET_FILES:
        _fetch(TRASHNET_BASE + name, out / name)
    if not (out / "dataset-resized").exists():
        with zipfile.ZipFile(out / "dataset-resized.zip") as zf:
            members = [m for m in zf.namelist() if not m.startswith("__MACOSX")]
            zf.extractall(out, members=members)
    return out


if __name__ == "__main__":
    download_rusentitweet()
    download_trashnet()
    print("Готово:", RAW)
