import argparse
from pathlib import Path
import json
import re
import math
import pandas as pd
import numpy as np
import urllib.parse as _up

LINK_DIR = Path("Exports/_links")
LINK_DIR.mkdir(parents=True, exist_ok=True)

RATES_SHEET_BASE = "Ставки (час–смена)"

INVALID_SHEET_CHARS_RE = re.compile(r"[\\/\*\?\:\[\]']")
ILLEGAL_XML_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")




def sanitize_excel_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет управляющие символы, которые ломают XML внутри XLSX."""
    out = df.copy()

    def _sanitize_value(value):
        if isinstance(value, str):
            return ILLEGAL_XML_CHARS_RE.sub("", value)
        return value

    out.columns = [ILLEGAL_XML_CHARS_RE.sub("", str(col)) for col in out.columns]
    object_cols = out.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        out[col] = out[col].map(_sanitize_value)
    return out

def sanitize_sheet_name(name: str, fallback: str = "Лист") -> str:
    s = (name or "").strip()
    s = INVALID_SHEET_CHARS_RE.sub("–", s)
    s = s.strip("'").strip()
    if not s:
        s = fallback
    if len(s) > 31:
        s = s[:31]
    return s


def unique_sheet_name(writer, desired: str) -> str:
    base = sanitize_sheet_name(desired)
    sheets = set(getattr(writer, "sheets", {}).keys())
    if base not in sheets:
        return base
    for i in range(2, 100):
        suffix = f" ({i})"
        trimmed = base[: max(0, 31 - len(suffix))] + suffix
        if trimmed not in sheets:
            return trimmed
    return sanitize_sheet_name(f"{base} ({len(sheets) + 1})")

def make_url_shortcut(url: str) -> Path:
    """Создаёт .url-ярлык и возвращает путь к нему."""
    safe = re.sub(r"[^0-9A-Za-z]+", "_", url)[-64:]
    p = LINK_DIR / f"{safe}.url"
    if not p.exists():
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"[InternetShortcut]\nURL={url}\n")
    return p

def _clean_url(u: str) -> str:
    if not u:
        return u
    try:
        p = _up.urlsplit(u.strip())
        scheme = p.scheme or "https"
        host = (p.netloc or "").lower()

        # убираем все ведущие www.
        while host.startswith("www."):
            host = host[4:]

        # m.avito.ru -> avito.ru
        if host.startswith("m.avito.ru"):
            host = "avito.ru"

        # итог: ровно один www для avito, иначе оставляем как есть
        netloc = "www.avito.ru" if host.endswith("avito.ru") else (p.netloc or "")

        # возвращаем без query/fragment
        return _up.urlunsplit((scheme, netloc, p.path, "", ""))
    except Exception:
        return u


EXPECTED_COLS = [
    "Должность","Работодатель","Дата публикации",
    "ЗП от (т.р.)","ЗП до (т.р.)",
    "Средний совокупный доход при графике 2/2 по 12 часов","В час","Длительность смены",
    "Требуемый опыт","Труд-во","График","Частота выплат","Льготы","Обязаности","Ссылка","Примечание"
]

SCHEDULE_SHIFTS_PER_MONTH = {
    "5/2": 22.0,
    "2/2": 15.0,
    "3/3": 15.0,
    "4/3": 17.0,
    "4/4": 15.0,
    "6/1": 26.0,
    "1/3": 8.0,
    "7/0": 30.0,
}

SCHEDULE_DEFAULT_HOURS = {
    "5/2": 8.0,
    "2/2": 12.0,
}

# ---- Helpers ----
def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx",".xls"]:
        return pd.read_excel(path)
    # CSV по умолчанию с запятой; если всё склеено — попробуем ; как запасной вариант
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # убрать \n, двойные пробелы, привести к нижнему для сопоставления
    cleaned = {c: re.sub(r"\s+", " ", str(c).replace("\\n", " ").replace("\n"," ")).strip() for c in df.columns}
    df = df.rename(columns=cleaned)

    # возможные варианты заголовков -> эталон
    mapping = {}
    for c in df.columns:
        x = c.lower()
        if x.startswith("должност"): mapping[c] = "Должность"
        elif x.startswith("работодат"): mapping[c] = "Работодатель"
        elif x.startswith("дата публи"): mapping[c] = "Дата публикации"
        elif "зп от" in x or "зарплата от" in x: mapping[c] = "ЗП от (т.р.)"
        elif "зп до" in x or "зарплата до" in x: mapping[c] = "ЗП до (т.р.)"
        elif "совокуп" in x or ("12" in x and "час" in x): mapping[c] = "Средний совокупный доход при графике 2/2 по 12 часов"
        elif x in ["в час","ставка в час","часовая ставка","ставка/час"]: mapping[c] = "В час"
        elif "длитель" in x and "смен" in x: mapping[c] = "Длительность смены"
        elif "требуем" in x and "опыт" in x: mapping[c] = "Требуемый опыт"
        elif "труд" in x: mapping[c] = "Труд-во"
        elif "график" in x and "доход" not in x: mapping[c] = "График"
        elif "частота" in x or "выплат" in x: mapping[c] = "Частота выплат"
        elif "льгот" in x: mapping[c] = "Льготы"
        elif "обязан" in x: mapping[c] = "Обязаности"
        elif "ссылка" in x or "url" in x: mapping[c] = "Ссылка"
        elif "примеч" in x or "коммент" in x: mapping[c] = "Примечание"

    df = df.rename(columns=mapping)

    # добавить отсутствующие столбцы
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # оставить и упорядочить только нужные
    df = df[EXPECTED_COLS]
    return df

def coerce_numbers(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["ЗП от (т.р.)","ЗП до (т.р.)","В час","Длительность смены","Средний совокупный доход при графике 2/2 по 12 часов"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(r"[^\d\.\-]", "", regex=True)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Если ставка в час отсутствует, считаем по формуле:
    #    (ЗП от / длительность смены / кол-во смен) × 1000
    #    ЗП от хранится в тысячах рублей.
    #    Если длительность смены не указана — берём дефолт по графику (только 5/2→8ч, 2/2→12ч).
    #    Если график не распознан — не рассчитываем.
    schedule_series = df.get("График", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str.strip()
    shifts_per_month = schedule_series.map(lambda s: SCHEDULE_SHIFTS_PER_MONTH.get(s, np.nan))

    hour_series = df.get("В час", pd.Series(np.nan, index=df.index))
    salary_from_series = df.get("ЗП от (т.р.)", pd.Series(np.nan, index=df.index))
    shift_length_series = df.get("Длительность смены", pd.Series(np.nan, index=df.index))

    # Fill missing shift length from schedule defaults (only 5/2 and 2/2)
    default_hours = schedule_series.map(lambda s: SCHEDULE_DEFAULT_HOURS.get(s, np.nan))
    effective_shift = shift_length_series.where(shift_length_series.notna() & (shift_length_series > 0), default_hours)

    mask_no_hour = (
        hour_series.isna()
        & salary_from_series.notna()
        & effective_shift.notna()
        & (effective_shift > 0)
        & shifts_per_month.notna()
        & (shifts_per_month > 0)
    )
    df.loc[mask_no_hour, "В час"] = (
        salary_from_series[mask_no_hour]
        / effective_shift[mask_no_hour]
        / shifts_per_month[mask_no_hour]
        * 1000.0
    )

    # Update shift length column with defaults used for calculation
    df.loc[mask_no_hour & shift_length_series.isna(), "Длительность смены"] = (
        default_hours[mask_no_hour & shift_length_series.isna()]
    )

    # 2) Рассчитываем "за 12 часов" только если "В час" определено.
    hour_series = df.get("В час", pd.Series(np.nan, index=df.index))
    df["Средний совокупный доход при графике 2/2 по 12 часов"] = np.where(
        hour_series.notna(),
        hour_series * 12.0,
        np.nan,
    )

    # 3) Если "Длительность смены" пусто — оставляем прочерк.
    if "Длительность смены" in df.columns:
        df["Длительность смены"] = df["Длительность смены"].astype(object)
        mask_len = df["Длительность смены"].isna()
        df.loc[mask_len, "Длительность смены"] = "-"

    # 4) Округление до 1 десятичного знака
    for col in ["В час", "Средний совокупный доход при графике 2/2 по 12 часов", "ЗП от (т.р.)", "ЗП до (т.р.)"]:
        if col in df.columns:
            df[col] = df[col].round(1)

    return df



def write_excel(df: pd.DataFrame, path: Path, rates: list[dict] | None = None):
    df = sanitize_excel_strings(df)
    # 1) Нормализуем колонку со ссылками заранее (если есть)
    if "Ссылка" in df.columns:
        try:
            df["Ссылка"] = (
                df["Ссылка"]
                .astype(str)
                .map(_clean_url)  # твоя функция: чистит m.avito.ru, www, убирает query/fragment
            )
        except Exception:
            pass

    # 2) Пишем файл. ВАЖНО: options -> через engine_kwargs
    with pd.ExcelWriter(
        path,
        engine="xlsxwriter",
        engine_kwargs={"options": {"strings_to_urls": False}},
    ) as xl:
        sheet = unique_sheet_name(xl, "Данные")
        df.to_excel(xl, sheet_name=sheet, index=False)
        ws = xl.sheets[sheet]

        # Freeze заголовки
        try:
            ws.freeze_panes(1, 0)
        except Exception:
            pass

        # Форматы
        fmt_link = xl.book.add_format({"underline": 1, "font_color": "blue"})
        fmt_num2 = xl.book.add_format({"num_format": "0.00"})
        fmt_int  = xl.book.add_format({"num_format": "0"})

        # 3) Явно проставим гиперссылки в колонке "Ссылка"
        if "Ссылка" in df.columns:
            try:
                col_idx = list(df.columns).index("Ссылка")
                # Переписываем ячейки как URL (иначе Excel не кликнет, т.к. strings_to_urls=False)
                for r, url in enumerate(df["Ссылка"].astype(str), start=1):  # row=1 — первая строка данных
                    u = (url or "").strip()
                    if u:
                        # нормализуем домен Avito и уберём query на всякий случай
                        u = re.sub(r"^https?://m\.avito\.ru", "https://www.avito.ru", u)
                        u = re.sub(r"^https?://avito\.ru", "https://www.avito.ru", u)
                        u = re.sub(r"\?.*$", "", u)
                    if u.startswith("http"):
                        ws.write_url(r, col_idx, u, fmt_link, u)
                    else:
                        ws.write(r, col_idx, u)
            except Exception:
                pass

        # 4) Применим числовые форматы к ключевым колонкам (если есть)
        num2_cols = [
            "В час",
            "Ставка (расчётная) в час, ₽",
            "Средний совокупный доход при графике 2/2 по 12 часов",
            "ЗП от (т.р.)",
            "ЗП до (т.р.)",
        ]
        for col in num2_cols:
            if col in df.columns:
                try:
                    cidx = list(df.columns).index(col)
                    # применим формат на разумный диапазон (первые 50k строк)
                    ws.set_column(cidx, cidx, None, fmt_num2)
                except Exception:
                    pass
        if "Длительность смены" in df.columns:
            try:
                cidx = list(df.columns).index("Длительность смены")
                ws.set_column(cidx, cidx, None, fmt_int)
            except Exception:
                pass

        # 5) Автоширина колонок (по первым ~200 строкам)
        for idx, col in enumerate(df.columns):
            try:
                sample = df[col].head(200).astype(str).tolist()
                max_len = max([len(str(col))] + [len(s) for s in sample])
            except Exception:
                max_len = len(str(col))
            ws.set_column(idx, idx, min(max_len + 2, 60))

        if rates:
            sheet_name = unique_sheet_name(xl, RATES_SHEET_BASE)
            rate_rows: list[dict[str, str]] = []
            for item in rates:
                if not isinstance(item, dict):
                    continue
                value = str(item.get("value") or "").strip()
                if not value:
                    continue
                rate_type = str(item.get("type") or "").lower()
                url_val = str(item.get("url") or "").strip()
                rate_rows.append({
                    "Ставка в час": value if rate_type == "hourly" else "",
                    "Ставка за смену": value if rate_type == "shift" else "",
                    "Ссылка": url_val,
                })

            if rate_rows:
                df_rates = pd.DataFrame(rate_rows, columns=["Ставка в час", "Ставка за смену", "Ссылка"])
                df_rates.to_excel(xl, sheet_name=sheet_name, index=False)
                ws_rates = xl.sheets[sheet_name]
                try:
                    ws_rates.freeze_panes(1, 0)
                except Exception:
                    pass
                try:
                    col_idx = 2
                    for ridx, link in enumerate(df_rates["Ссылка"].astype(str), start=1):
                        u = (link or "").strip()
                        if u.startswith("http"):
                            ws_rates.write_url(ridx, col_idx, u, fmt_link, u)
                        else:
                            ws_rates.write(ridx, col_idx, u)
                except Exception:
                    pass
                try:
                    ws_rates.set_column(0, 1, 26)
                    ws_rates.set_column(2, 2, 60)
                except Exception:
                    pass



def main():
    ap = argparse.ArgumentParser(description="Строит чистый XLSX из сырых данных парсера")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    df = load_any(src)
    df = normalize_columns(df)
    df = coerce_numbers(df)
    df = compute_metrics(df)

    rates_path = Path(str(src) + ".rates.json")
    rates_data: list[dict] | None = None
    if rates_path.exists():
        try:
            with open(rates_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("items"), list):
                rates_data = [item for item in data.get("items", []) if isinstance(item, dict)]
            elif isinstance(data, list):
                rates_data = [item for item in data if isinstance(item, dict)]
        except Exception:
            rates_data = None

    write_excel(df, dst, rates=rates_data)
    print(f"Готово: {dst} ({len(df)} строк)")

if __name__ == "__main__":
    main()
