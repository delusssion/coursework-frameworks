"""Ручное обогащение реальными данными: отбор и подготовка новых объектов.

Тексты (к RuSentiTweet) — реальные сообщения пользователей русскоязычных
интернет-платформ из открытого корпуса Skoltech «Inappropriate messages»
(Babakov et al., 2021; github.com/skoltech-nlp/inappropriate-sensitive-topics,
лицензия CC BY-NC-SA 4.0). Берём только сообщения без чувствительной темы
(колонка none = 1), 4–30 слов, которых нет в RuSentiTweet. Исходные метки
корпуса (тема, «неприемлемость») к тональности не относятся — тональность
размечаем сами по codebook. 40 сообщений берутся случайно (реальное
распределение источника), 30 — целевым отбором по словам-маркерам редких
классов (благодарность, поздравление, восторг): случайная выборка из
форумного источника почти целиком нейтральна. Колонка sampling служебная.

Фото (к TrashNet) — реальные фотографии отходов из TrashBox
(Kumsetty et al., 2022; github.com/nikhilvenkatkumsetty/TrashBox). Сняты в
разных условиях (фон, освещение, ракурс), в отличие от белого фона TrashNet.
Подклассы отобраны так, чтобы покрыть все 6 классов TrashNet, включая trash
(тетрапак, бумажные стаканы, окурки, перчатки, маски). Класс по нашему
codebook определяют разметчики: метка TrashBox сохраняется только в
служебном файле и в листы разметки не попадает. TrashBox собран из интернета
и содержит копии фото TrashNet — такие фото (pHash-расстояние ≤ 4 до любого
фото TrashNet) отбрасываются, список — images_rejected_trashnet_copies.csv.

Запуск:  python -m src.collect_real
Результат:
  data/enriched/real/texts.csv            id, text (+ служебные поля)
  data/enriched/real/images/*.jpg         фото, ≤512 px, без EXIF
  data/enriched/real/images_manifest.csv  id, файл, источник, условия
"""
from __future__ import annotations

import hashlib
import io
import random
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "enriched" / "real"
CACHE = ROOT / "data" / "cache"
SEED = 42

TEXT_URL = ("https://raw.githubusercontent.com/skoltech-nlp/inappropriate-sensitive-topics/"
            "main/Version1/appropriateness/test.csv")
N_RANDOM = 40      # случайная выборка — реальное распределение источника
N_TARGETED = 30    # целевой отбор кандидатов в редкие классы (speech, positive)
TARGET_RE = re.compile(
    r"спасиб|благодар|поздравл|с днём|с днем|с праздник|привет|доброе утро|доброй ночи|спокойной ночи|"
    r"люблю|обожаю|класс|круто|отличн|супер|ура\b|прекрасн|счастл|молодец|здорово|восхит|красив|удачи", re.I)

TRASHBOX_GIT = "https://github.com/nikhilvenkatkumsetty/TrashBox"
TRASHBOX_RAW = "https://raw.githubusercontent.com/nikhilvenkatkumsetty/TrashBox/main/"
TB_DIR = "TrashBox_train_dataset_subfolders"
# (класс TrashBox, подкласс TrashBox) -> сколько фото взять
IMAGE_PLAN = {
    ("cardboard", None): 10,
    ("paper", "news paper"): 3, ("paper", "paper"): 4, ("paper", "paper cups"): 3, ("paper", "tetra pak"): 4,
    ("glass", None): 10,
    ("metal", "beverage cans"): 4, ("metal", "metal containers"): 3, ("metal", "spray cans"): 3,
    ("plastic", "plastic bottles"): 3, ("plastic", "plastic containers"): 3, ("plastic", "plastic bags"): 3,
    ("plastic", "plastic cups"): 2, ("plastic", "cigarette butt"): 2,
    ("medical", "gloves"): 3, ("medical", "masks"): 3,
}

RE_WORD = re.compile(r"[а-яёa-z]+", re.I)


def mask_pii(s: str) -> str:
    s = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[email]", s)
    s = re.sub(r"(?<![\w@])@[A-Za-z0-9_]{3,}", "@user", s)
    s = re.sub(r"(\+7|\b8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}\b", "[телефон]", s)
    s = re.sub(r"https?://\S+", "[ссылка]", s)
    return re.sub(r"\s+", " ", s).strip()


def norm(s: str) -> str:
    return " ".join(RE_WORD.findall(re.sub(r"https?://\S+|@\w+", " ", str(s).lower())))


def collect_texts() -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    src = CACHE / "skoltech_appropriateness_test.csv"
    if not src.exists():
        urllib.request.urlretrieve(TEXT_URL, src)
    d = pd.read_csv(src)
    d = d[d["none"] == 1].copy()                         # без чувствительной темы
    d["text"] = d["text"].astype(str).map(mask_pii)
    d["n_words"] = d["text"].map(lambda s: len(RE_WORD.findall(s)))
    d = d[(d["n_words"] >= 4) & (d["n_words"] <= 30)]
    seen = set(pd.read_csv(ROOT / "data/raw/rusentitweet/rusentitweet_full.csv")["text"].map(norm))
    d["norm"] = d["text"].map(norm)
    d = d[~d["norm"].isin(seen)].drop_duplicates("norm")
    hit = d["text"].str.contains(TARGET_RE)
    targeted = d[hit].sample(N_TARGETED, random_state=SEED).assign(sampling="targeted")
    rand = d[~d.index.isin(targeted.index)].sample(N_RANDOM, random_state=SEED).assign(sampling="random")
    # перемешиваем, чтобы разметчики не видели, какие объекты отобраны целевым образом
    d = pd.concat([rand, targeted]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    out = pd.DataFrame({
        "id": [f"t{i + 1:03d}" for i in range(len(d))],
        "text": d["text"],
        "source": "skoltech-nlp/inappropriate-sensitive-topics, Version1/appropriateness/test.csv",
        "sampling": d["sampling"],                       # служебно: random / targeted
        "source_inappropriate": d["inappropriate"],      # служебно, к тональности не относится
    })
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "texts.csv", index=False)
    return out


def _trashbox_tree() -> list[str]:
    repo = CACHE / "TrashBox"
    if not repo.exists():
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", "--depth", "1",
                        TRASHBOX_GIT, str(repo)], check=True)
    res = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "HEAD", "--name-only", TB_DIR],
                         check=True, capture_output=True, text=True)
    return res.stdout.splitlines()


def _trashnet_hashes() -> list:
    """pHash всех фото TrashNet: TrashBox собран из интернета и содержит копии TrashNet."""
    root = ROOT / "data/raw/trashnet/dataset-resized"
    return [imagehash.phash(Image.open(f)) for f in sorted(root.glob("*/*.jpg"))]


def collect_images() -> pd.DataFrame:
    tree = _trashbox_tree()
    rng = random.Random(SEED)
    tn_hashes = _trashnet_hashes()
    img_dir = OUT / "images"
    if img_dir.exists():
        for f in img_dir.glob("*.jpg"):
            f.unlink()
    img_dir.mkdir(parents=True, exist_ok=True)
    rows, k, leaked = [], 0, []
    for (cls, sub), n in IMAGE_PLAN.items():
        prefix = f"{TB_DIR}/{cls}/" + (f"{sub}/" if sub else "")
        cands = sorted(p for p in tree if p.startswith(prefix) and p.lower().endswith((".jpg", ".jpeg", ".png")))
        rng.shuffle(cands)
        taken = 0
        for path in cands:
            if taken == n:
                break
            raw = urllib.request.urlopen(TRASHBOX_RAW + urllib.parse.quote(path), timeout=60).read()
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
            h = imagehash.phash(im)
            if min(h - t for t in tn_hashes) <= 4:      # копия фото из TrashNet — утечка, пропускаем
                leaked.append(path)
                continue
            taken += 1
            w0, h0 = im.size
            im.thumbnail((512, 512))
            k += 1
            iid = f"i{k:03d}"
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=90)                # пересохранение без EXIF
            (img_dir / f"{iid}.jpg").write_bytes(buf.getvalue())
            rows.append({"id": iid, "file": f"images/{iid}.jpg", "width": im.width, "height": im.height,
                         "orig_width": w0, "orig_height": h0,
                         "source": "TrashBox (nikhilvenkatkumsetty/TrashBox)", "source_path": path,
                         "source_class": cls, "source_subclass": sub or cls,
                         "md5": hashlib.md5(buf.getvalue()).hexdigest()})
    # перемешиваем порядок, чтобы разметчики не видели групп по классам
    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df["order"] = range(1, len(df) + 1)
    df.to_csv(OUT / "images_manifest.csv", index=False)
    pd.DataFrame({"source_path": leaked}).to_csv(OUT / "images_rejected_trashnet_copies.csv", index=False)
    print(f"отброшено копий TrashNet: {len(leaked)}")
    return df


if __name__ == "__main__":
    t = collect_texts()
    print("тексты:", len(t))
    im = collect_images()
    print("фото:", len(im), im["source_subclass"].value_counts().to_dict())
