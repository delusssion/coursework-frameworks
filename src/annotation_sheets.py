"""Листы разметки: отдельный Excel-файл на каждого разметчика (слепая разметка).

Запуск:  python -m src.annotation_sheets
Результат: data/annotation/sheets/разметка_<Фамилия>.xlsx

Разметчик заполняет только колонки «Метка», «Сомневаюсь», «Комментарий»
на листах «Тексты» и «Фото». Метки выбираются из выпадающего списка.
Классы источника в листы не попадают.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "enriched" / "real"
OUT = ROOT / "data" / "annotation" / "sheets"

ANNOTATORS = ["Шалыгин Ярослав", "Соколов Александр"]
CAL_N = 20

TEXT_LABELS = [
    ("positive", "Позитив", "явная или неявная положительная эмоция/оценка; желание хорошего; хороший опыт"),
    ("negative", "Негатив", "отрицательная эмоция/оценка; сарказм; «не хочу»; плохой опыт; «скучаю»"),
    ("neutral", "Нейтрально", "факт, вопрос, совет, новость, «да/нет/ок»; смайлики — единственный признак эмоции"),
    ("speech", "Речевой акт", "главное — спасибо, привет, поздравление, пожелание, извинение"),
    ("skip", "Пропуск", "тональность неясна, нужен контекст, шутка/стихи, не русский, спам"),
]
IMG_LABELS = [
    ("cardboard", "Картон", "гофрокартон, коробки, яичные лотки, прессованный картон"),
    ("paper", "Бумага", "листы, газеты, журналы, бумажные пакеты, чеки"),
    ("glass", "Стекло", "бутылки, банки, флаконы (не керамика, не украшения)"),
    ("metal", "Металл", "банки, жесть, баллончики, крышки, фольга"),
    ("plastic", "Пластик", "ПЭТ, флаконы, контейнеры, пакеты, пластиковые стаканы"),
    ("trash", "Несортируемое", "тетрапак, бумажные стаканы, окурки, маски, перчатки, грязное"),
    ("exclude", "Исключить", "рисунок/постер, не отход, несколько разных материалов, не видно предмет"),
]

FONT = "Arial"
HEAD_FILL = PatternFill("solid", fgColor="2F6B4F")
INPUT_FILL = PatternFill("solid", fgColor="FFF6CC")   # ячейки для заполнения
CAL_FILL = PatternFill("solid", fgColor="EDEFE9")
THIN = Border(bottom=Side(style="thin", color="D6DAD2"))


def _style_header(ws, row: int, ncols: int) -> None:
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF")
        cell.fill = HEAD_FILL
        cell.alignment = Alignment(vertical="center", wrap_text=True)


def _add_validation(ws, labels, col: str, first: int, last: int) -> None:
    dv = DataValidation(type="list", formula1='"' + ",".join(l[0] for l in labels) + '"', allow_blank=True,
                        showErrorMessage=True, errorTitle="Неизвестная метка",
                        error="Выберите метку из списка (см. лист «Шпаргалка»).")
    ws.add_data_validation(dv)
    dv.add(f"{col}{first}:{col}{last}")
    dv2 = DataValidation(type="list", formula1='"да"', allow_blank=True)
    ws.add_data_validation(dv2)
    dv2.add(f"{chr(ord(col) + 1)}{first}:{chr(ord(col) + 1)}{last}")


def _sheet_texts(wb, texts: pd.DataFrame) -> None:
    ws = wb.create_sheet("Тексты")
    ws["A1"] = "Тексты → RuSentiTweet"
    ws["A1"].font = Font(name=FONT, bold=True, size=13)
    ws["D1"] = "Размечено:"
    ws["E1"] = f'=COUNTA(D4:D{3 + len(texts)})&" из {len(texts)}"'
    for c in ("D1", "E1"):
        ws[c].font = Font(name=FONT, bold=True)
    ws["A2"] = "Заполняйте только жёлтые колонки. Метка — из выпадающего списка. Первые 20 строк (серые) — калибровка."
    ws["A2"].font = Font(name=FONT, italic=True, color="5D6459")
    heads = ["№", "id", "Сообщение", "Метка", "Сомневаюсь", "Комментарий"]
    ws.append([])
    for i, h in enumerate(heads, 1):
        ws.cell(row=3, column=i, value=h)
    _style_header(ws, 3, len(heads))
    for i, r in enumerate(texts.itertuples(), 1):
        row = 3 + i
        vals = [i, r.id, r.text, None, None, None]
        for c, v in enumerate(vals, 1):
            cell = ws.cell(row=row, column=c, value=v)
            cell.font = Font(name=FONT, size=11 if c == 3 else 10)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.border = THIN
            if c >= 4:
                cell.fill = INPUT_FILL
            elif i <= CAL_N:
                cell.fill = CAL_FILL
    _add_validation(ws, TEXT_LABELS, "D", 4, 3 + len(texts))
    for col, w in zip("ABCDEF", [5, 7, 80, 14, 12, 30]):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "D4"


def _sheet_images(wb, imgs: pd.DataFrame) -> None:
    ws = wb.create_sheet("Фото")
    ws["A1"] = "Фото → TrashNet"
    ws["A1"].font = Font(name=FONT, bold=True, size=13)
    ws["D1"] = "Размечено:"
    ws["E1"] = f'=COUNTA(D4:D{3 + len(imgs)})&" из {len(imgs)}"'
    for c in ("D1", "E1"):
        ws[c].font = Font(name=FONT, bold=True)
    ws["A2"] = "Определите МАТЕРИАЛ предмета по codebook. Первые 20 строк — калибровка."
    ws["A2"].font = Font(name=FONT, italic=True, color="5D6459")
    heads = ["№", "id", "Фото", "Метка", "Сомневаюсь", "Комментарий"]
    for i, h in enumerate(heads, 1):
        ws.cell(row=3, column=i, value=h)
    _style_header(ws, 3, len(heads))
    for i, r in enumerate(imgs.itertuples(), 1):
        row = 3 + i
        for c, v in enumerate([i, r.id, None, None, None, None], 1):
            cell = ws.cell(row=row, column=c, value=v)
            cell.font = Font(name=FONT, size=10)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = THIN
            if c >= 4:
                cell.fill = INPUT_FILL
            elif i <= CAL_N:
                cell.fill = CAL_FILL
        im = Image.open(REAL / r.file)
        im.thumbnail((240, 180))
        buf = io.BytesIO()
        im.save(buf, "PNG")
        buf.seek(0)
        xl = XLImage(buf)
        ws.add_image(xl, f"C{row}")
        ws.row_dimensions[row].height = 140
    _add_validation(ws, IMG_LABELS, "D", 4, 3 + len(imgs))
    for col, w in zip("ABCDEF", [5, 7, 36, 14, 12, 30]):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "D4"


def _sheet_help(wb, annotator: str) -> None:
    ws = wb.active
    ws.title = "Инструкция"
    lines = [
        (f"Лист разметки: {annotator}", True),
        ("", False),
        ("1. Прочитайте codebook (docs/codebook_text.md и docs/codebook_images.md) и лист «Шпаргалка».", False),
        ("2. Размечайте САМОСТОЯТЕЛЬНО: не смотрите и не обсуждайте метки второго разметчика до конца разметки.", False),
        ("3. Сначала первые 20 строк на каждом листе (калибровка, серые) — затем сверка с напарником и при необходимости уточнение codebook.", False),
        ("4. Затем остальные строки. Заполняйте только жёлтые колонки:", False),
        ("   • Метка — выберите из выпадающего списка;", False),
        ("   • Сомневаюсь — «да», если выбор дался с трудом (если тональность совсем неясна — это метка skip);", False),
        ("   • Комментарий — по желанию, почему случай сложный.", False),
        ("5. Сохраните файл и положите в data/annotation/ под тем же именем.", False),
        ("", False),
        ("Пример заполненной строки (Тексты):", True),
        ("№ 1 | t001 | «Вау, спасибо. Тогда на днях вернусь за ним.» | speech | да | благодарность + радость, по codebook главное — благодарность", False),
    ]
    for i, (t, b) in enumerate(lines, 1):
        c = ws.cell(row=i, column=1, value=t)
        c.font = Font(name=FONT, bold=b, size=13 if i == 1 else 11)
    ws.column_dimensions["A"].width = 130


def _sheet_cheat(wb) -> None:
    ws = wb.create_sheet("Шпаргалка")
    ws.append(["Датасет", "Метка", "Название", "Когда ставить"])
    _style_header(ws, 1, 4)
    for ds, labels in (("Тексты", TEXT_LABELS), ("Фото", IMG_LABELS)):
        for lid, name, hint in labels:
            ws.append([ds, lid, name, hint])
    ws.append([])
    ws.append(["Тексты", "порядок", "", "1) понятна ли тональность? нет → skip; 2) главное — ритуальная формула? → speech; 3) есть оценка/эмоция? → positive/negative; 4) иначе neutral"])
    ws.append(["Фото", "порядок", "", "1) фото пригодно и это отход? нет → exclude; 2) составной/грязный (тетрапак, бум. стакан, окурок, маска)? → trash; 3) иначе материал корпуса"])
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.font = Font(name=FONT, size=10)
            c.alignment = Alignment(wrap_text=True, vertical="top")
    for col, w in zip("ABCD", [10, 12, 16, 100]):
        ws.column_dimensions[col].width = w


def build() -> list[Path]:
    texts = pd.read_csv(REAL / "texts.csv")
    imgs = pd.read_csv(REAL / "images_manifest.csv").sort_values("order")
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ANNOTATORS:
        wb = Workbook()
        _sheet_help(wb, name)
        _sheet_texts(wb, texts)
        _sheet_images(wb, imgs)
        _sheet_cheat(wb)
        p = OUT / f"разметка_{name.split()[0]}.xlsx"
        wb.save(p)
        paths.append(p)
    return paths


if __name__ == "__main__":
    for p in build():
        print(p.relative_to(ROOT))
