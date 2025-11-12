"""Pipeline helpers for collecting Avito vacancies and exporting analytics.

The module provides two main entry points used by the CLI in ``run_pipeline``:

``fetch_avito_vacancies``
    Collects vacancy records from Avito with the help of :class:`AvitoCollector`
    and transforms them into a normalized tabular representation that follows
    the specification from the task description.

``export_avito_to_excel``
    Builds an Excel workbook with two analytical sheets (data & analytics) and a
    log sheet containing diagnostic information about the collection process.

The implementation is intentionally deterministic so it can be tested with the
existing HTML fixtures that power :mod:`tests.test_avito_source`.  Any
assumptions made during salary normalisation are recorded in the
``Примечания расчета`` column and mirrored in the diagnostic log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from avito_source import AvitoConfig, collect_avito_vacancies


DEFAULT_OUTPUT_PATH = Path("Exports/vacancies_avito.xlsx")


# ---------------------------------------------------------------------------
# Configuration


def _ensure_list(value: Optional[Iterable[str] | str]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = [chunk.strip() for chunk in value.split(",")]
    else:
        raw = [str(chunk).strip() for chunk in value]
    return [chunk for chunk in raw if chunk]


def _load_schedule_defaults(raw: Optional[dict]) -> Dict[str, Dict[str, float]]:
    defaults: Dict[str, Dict[str, float]] = {
        "5/2": {"shifts_per_month": 23.0, "hours_per_shift": 8.0},
        "2/2": {"shifts_per_month": 15.0, "hours_per_shift": 12.0},
        "3/3": {"shifts_per_month": 15.0, "hours_per_shift": 12.0},
        "6/1": {"shifts_per_month": 26.0, "hours_per_shift": 8.0},
    }
    if not raw:
        return defaults
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        hours = float(value.get("hours_per_shift", defaults.get(key, {}).get("hours_per_shift", 0)))
        shifts = float(value.get("shifts_per_month", defaults.get(key, {}).get("shifts_per_month", 0)))
        if hours <= 0:
            hours = defaults.get(key, {}).get("hours_per_shift", 0) or 0
        if shifts <= 0:
            shifts = defaults.get(key, {}).get("shifts_per_month", 0) or 0
        if hours > 0 or shifts > 0:
            defaults[key] = {"hours_per_shift": hours, "shifts_per_month": shifts}
    return defaults


@dataclass
class AvitoPipelineConfig:
    """Top-level configuration used by the Avito pipeline."""

    queries: List[str] = field(default_factory=list)
    city: str = ""
    region: str = ""
    min_salary: Optional[float] = None
    max_salary: Optional[float] = None
    date_from_days: Optional[int] = None
    pages_limit: int = 3
    headless: bool = True
    schedule_defaults: Dict[str, Dict[str, float]] = field(default_factory=dict)
    delay_range: Tuple[float, float] = (1.5, 4.0)

    @classmethod
    def from_mapping(cls, payload: Dict[str, object]) -> "AvitoPipelineConfig":
        queries = _ensure_list(payload.get("query"))
        city = str(payload.get("city", "") or "").strip()
        region = str(payload.get("region", "") or "").strip() or city
        pages_limit = int(payload.get("pages_limit", 3) or 3)
        date_from = payload.get("date_from_days")
        min_salary = payload.get("min_salary")
        max_salary = payload.get("max_salary")
        headless = bool(payload.get("headless", True))
        delay_range = payload.get("delay_range") or (1.5, 4.0)
        if isinstance(delay_range, (list, tuple)) and len(delay_range) == 2:
            lo = max(0.3, float(delay_range[0]))
            hi = max(lo, float(delay_range[1]))
            delay_range = (lo, hi)
        else:
            delay_range = (1.5, 4.0)

        def _coerce_float(value: Optional[object]) -> Optional[float]:
            if value in (None, "", "null"):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        schedule_defaults = _load_schedule_defaults(payload.get("graph_schedule_defaults"))

        date_from_value = _coerce_float(date_from)

        return cls(
            queries=queries,
            city=city,
            region=region or city,
            min_salary=_coerce_float(min_salary),
            max_salary=_coerce_float(max_salary),
            date_from_days=int(date_from_value) if date_from_value is not None else None,
            pages_limit=max(1, pages_limit),
            headless=headless,
            schedule_defaults=schedule_defaults,
            delay_range=delay_range,  # type: ignore[arg-type]
        )


def load_config(path: Path) -> AvitoPipelineConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):  # pragma: no cover - defensive
        raise ValueError("Config must be a JSON object")
    return AvitoPipelineConfig.from_mapping(data)


# ---------------------------------------------------------------------------
# Normalisation helpers


_WS_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_EMOJI_RE = re.compile(r"[\U00010000-\U0010FFFF]")


def _clean_text(value: Optional[str]) -> str:
    if not value:
        return ""
    text = str(value)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _EMOJI_RE.sub("", text)
    text = text.replace("\u202f", " ").replace("\xa0", " ")
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _normalize_experience(raw: Optional[str]) -> str:
    text = _clean_text(raw).lower()
    if not text:
        return ""
    if "без опыта" in text or "не требуется" in text:
        return "без опыта"
    if re.search(r"(1|один).*?([23]|тр)[^\d]*год", text):
        return "1-3"
    if re.search(r"(3|три).*?6", text) or "3-6" in text:
        return "3-6"
    if re.search(r"(6|шест).*(лет|год)", text):
        return "6+"
    if "от 1" in text:
        return "1-3"
    if "от 3" in text:
        return "3-6"
    if "от 5" in text or "от 6" in text:
        return "6+"
    return text


def _normalize_schedule(raw: Optional[str]) -> str:
    text = _clean_text(raw)
    if not text:
        return ""
    text_low = text.lower()
    if "вахт" in text_low:
        return "вахта"
    match = re.search(r"(\d+\s*/\s*\d+)", text_low)
    if match:
        return match.group(1).replace(" ", "")
    if "гибк" in text_low:
        return "гибкий"
    if "смен" in text_low and "сут" in text_low:
        return "сутки/сутки"
    return text


def _infer_format_work(text_chunks: Iterable[str], schedule: str) -> str:
    blob = " ".join(_clean_text(chunk).lower() for chunk in text_chunks if chunk)
    if "удал" in blob or "home office" in blob or "remote" in blob:
        return "удаленно"
    if "гибрид" in blob or "частично удал" in blob:
        return "гибрид"
    if "вахт" in blob or schedule.lower() == "вахта":
        return "вахта"
    if "разъезд" in blob:
        return "разъездная"
    if "офис" in blob or "на территории" in blob or "в офисе" in blob:
        return "офис"
    return ""


def _split_benefits(text: str) -> List[str]:
    if not text:
        return []
    parts = [chunk.strip(" ;,•-") for chunk in text.split(";")]
    items = []
    for part in parts:
        if not part:
            continue
        if "," in part and ";" not in part:
            subparts = [sub.strip() for sub in part.split(",") if sub.strip()]
            if len(subparts) > 1:
                items.extend(subparts)
                continue
        items.append(part)
    return [item for item in items if item]


def _join_unique(items: Iterable[str]) -> str:
    seen = set()
    result: List[str] = []
    for item in items:
        cleaned = _clean_text(item)
        if not cleaned:
            continue
        if cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        result.append(cleaned)
    return "; ".join(result)


@dataclass
class SalaryResult:
    salary_from_thousands: Optional[float]
    salary_to_thousands: Optional[float]
    hourly_rate: Optional[float]
    shift_income: Optional[float]
    notes: List[str]


def _derive_shift_params(record: Dict[str, object], config: AvitoPipelineConfig) -> Tuple[str, Optional[float], Optional[float], List[str]]:
    notes: List[str] = []
    schedule = _normalize_schedule(record.get("schedule_hint") or "")
    working_hours = record.get("working_hours") or {}
    if isinstance(working_hours, dict):
        norm = working_hours.get("normalized") if isinstance(working_hours.get("normalized"), dict) else {}
        shift_based = norm.get("shift_based") if isinstance(norm, dict) else {}
        if isinstance(shift_based, dict):
            if not schedule and shift_based.get("pattern"):
                schedule = _normalize_schedule(str(shift_based.get("pattern")))
            hours = shift_based.get("shift_length_hours") or shift_based.get("hours_per_shift")
            if hours:
                try:
                    hours_float = float(hours)
                    if hours_float > 0:
                        notes.append(f"часы/смену={hours_float:g} (из normalized)")
                        hours_per_shift = hours_float
                    else:
                        hours_per_shift = None
                except (TypeError, ValueError):
                    hours_per_shift = None
            else:
                hours_per_shift = None
        else:
            hours_per_shift = None
    else:
        norm = {}
        hours_per_shift = None

    shifts_per_month: Optional[float] = None

    if not hours_per_shift:
        raw_matches = []
        if isinstance(working_hours, dict):
            raw_matches = working_hours.get("raw_matches") or []
        for match in raw_matches or []:
            cleaned = _clean_text(match)
            m = re.search(r"(\d{1,2})\s*(?:час|ч)", cleaned.lower())
            if m:
                hours_per_shift = float(m.group(1))
                notes.append(f"часы/смену={hours_per_shift:g} (из текста)")
                break

    if schedule:
        defaults = config.schedule_defaults.get(schedule)
        if defaults:
            if not hours_per_shift and defaults.get("hours_per_shift"):
                hours_per_shift = float(defaults["hours_per_shift"])
                notes.append(f"часы/смену={hours_per_shift:g} (по умолчанию)")
            if defaults.get("shifts_per_month"):
                shifts_per_month = float(defaults["shifts_per_month"])
                notes.append(f"смен/мес={shifts_per_month:g} (по умолчанию)")
    else:
        schedule = ""

    return schedule, hours_per_shift, shifts_per_month, notes


def _convert_salary(
    record: Dict[str, object],
    config: AvitoPipelineConfig,
    schedule: str,
    hours_per_shift: Optional[float],
    shifts_per_month: Optional[float],
    assumption_notes: List[str],
) -> SalaryResult:
    notes = list(assumption_notes)
    salary_from = record.get("salary_from")
    salary_to = record.get("salary_to")
    period = (record.get("salary_period") or "per_month").lower()
    currency = (record.get("salary_currency") or "RUB").upper()

    def _as_float(value: object) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    salary_from_val = _as_float(salary_from)
    salary_to_val = _as_float(salary_to)

    if salary_from_val is not None and salary_to_val is not None and salary_from_val > salary_to_val:
        salary_from_val, salary_to_val = salary_to_val, salary_from_val
        notes.append("ЗП от > ЗП до — значения поменяны местами")

    if currency and currency != "RUB":
        notes.append(f"Валюта {currency} не поддерживается")
        return SalaryResult(None, None, None, None, notes)

    salary_from_thousands: Optional[float] = None
    salary_to_thousands: Optional[float] = None
    hourly_rate: Optional[float] = None
    shift_income: Optional[float] = None

    def _ensure_defaults() -> Tuple[Optional[float], Optional[float]]:
        return hours_per_shift, shifts_per_month

    if period in {"per_month", "month", "monthly"}:
        if salary_from_val is not None:
            salary_from_thousands = salary_from_val / 1000.0
        if salary_to_val is not None:
            salary_to_thousands = salary_to_val / 1000.0
    elif period in {"per_shift", "shift"}:
        hours_per_shift, shifts_per_month = _ensure_defaults()
        if shifts_per_month and salary_from_val is not None:
            salary_from_thousands = (salary_from_val * shifts_per_month) / 1000.0
        if shifts_per_month and salary_to_val is not None:
            salary_to_thousands = (salary_to_val * shifts_per_month) / 1000.0
        if salary_from_val and not shift_income:
            shift_income = salary_from_val
    elif period in {"per_hour", "hour"}:
        hours_per_shift, shifts_per_month = _ensure_defaults()
        if shifts_per_month and hours_per_shift and salary_from_val is not None:
            salary_from_thousands = (salary_from_val * hours_per_shift * shifts_per_month) / 1000.0
        if shifts_per_month and hours_per_shift and salary_to_val is not None:
            salary_to_thousands = (salary_to_val * hours_per_shift * shifts_per_month) / 1000.0
        if salary_from_val is not None:
            hourly_rate = salary_from_val
        elif salary_to_val is not None:
            notes.append("Ставка в час дана только максимумом — расчёт пропущен")
    elif period in {"per_day", "day"}:
        hours_per_shift, shifts_per_month = _ensure_defaults()
        if shifts_per_month and salary_from_val is not None:
            salary_from_thousands = (salary_from_val * shifts_per_month) / 1000.0
            shift_income = salary_from_val
        if shifts_per_month and salary_to_val is not None:
            salary_to_thousands = (salary_to_val * shifts_per_month) / 1000.0
    elif period in {"per_week", "week"}:
        hours_per_shift, shifts_per_month = _ensure_defaults()
        weeks_per_month = 4.33
        if salary_from_val is not None:
            salary_from_thousands = (salary_from_val * weeks_per_month) / 1000.0
        if salary_to_val is not None:
            salary_to_thousands = (salary_to_val * weeks_per_month) / 1000.0

    if salary_from_thousands is not None and salary_from_thousands < 0:
        salary_from_thousands = None
        notes.append("ЗП от < 0 — очищено")
    if salary_to_thousands is not None and salary_to_thousands < 0:
        salary_to_thousands = None
        notes.append("ЗП до < 0 — очищено")

    if salary_from_thousands is not None and hours_per_shift and shifts_per_month:
        hourly_rate = (salary_from_thousands * 1000.0) / (hours_per_shift * shifts_per_month)

    if hourly_rate is not None:
        shift_income = hourly_rate * 12.0
    elif shift_income is None and salary_from_val and period in {"per_shift", "per_day"}:
        shift_income = salary_from_val

    return SalaryResult(salary_from_thousands, salary_to_thousands, hourly_rate, shift_income, notes)


def _extract_working_hours_text(record: Dict[str, object]) -> str:
    working_hours = record.get("working_hours")
    if isinstance(working_hours, dict):
        matches = working_hours.get("raw_matches")
        if matches:
            return _join_unique(matches)
    return ""


def _pick_region(record: Dict[str, object], config: AvitoPipelineConfig) -> str:
    region = _clean_text(record.get("location_area") or record.get("region"))
    if region:
        return region
    return config.region or config.city


def _published_date(record: Dict[str, object]) -> Optional[date]:
    posted_at = record.get("posted_at")
    if isinstance(posted_at, datetime):
        return posted_at.date()
    raw = _clean_text(record.get("posted_at_raw"))
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
        return parsed.date()
    except ValueError:
        return None


def _collect_conditions(record: Dict[str, object]) -> str:
    benefits = record.get("benefits") or []
    benefit_text = "; ".join(benefits) if isinstance(benefits, (list, tuple)) else ""
    conditions_raw = _clean_text(record.get("conditions_raw"))
    parts: List[str] = []
    if conditions_raw:
        parts.extend(part.strip() for part in re.split(r"[\n;]", conditions_raw) if part.strip())
    if benefit_text:
        parts.extend(part.strip() for part in benefit_text.split(";") if part.strip())
    return _join_unique(parts)


def _normalize_record(record: Dict[str, object], config: AvitoPipelineConfig, collected_at: date) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    schedule, hours_per_shift, shifts_per_month, notes = _derive_shift_params(record, config)
    salary = _convert_salary(record, config, schedule, hours_per_shift, shifts_per_month, notes)

    city = _clean_text(record.get("location_city") or config.city)
    region = _pick_region(record, config)
    title = _clean_text(record.get("title"))
    company = _clean_text(record.get("employer_name")) or "Не указано"
    duties = _clean_text(record.get("duties_raw"))
    requirements = _clean_text(record.get("requirements_raw"))
    contact_raw = _clean_text((record.get("contact") or {}).get("raw") if isinstance(record.get("contact"), dict) else None)
    working_hours_text = _extract_working_hours_text(record)
    experience_text = _normalize_experience((record.get("experience_required") or {}).get("raw") if isinstance(record.get("experience_required"), dict) else record.get("experience_required"))
    employment = _clean_text(record.get("employment_type"))
    format_work = _infer_format_work(
        [employment, duties, requirements, record.get("conditions_raw"), working_hours_text],
        schedule,
    )
    conditions = _collect_conditions(record)
    published = _published_date(record)

    calc_notes = salary.notes
    if schedule:
        marker = f"График={schedule}"
        if marker not in calc_notes:
            calc_notes.append(marker)
    if hours_per_shift:
        marker = f"часы/смену={hours_per_shift:g}"
        if marker not in calc_notes:
            calc_notes.append(marker)
    if shifts_per_month:
        marker = f"смен/мес={shifts_per_month:g}"
        if marker not in calc_notes:
            calc_notes.append(marker)

    logs: List[Dict[str, object]] = []
    if record.get("is_active") is False:
        logs.append(
            {
                "stage": "card",
                "message": "Объявление недоступно",
                "url": record.get("url_detail"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    if salary.notes:
        logs.append(
            {
                "stage": "calc",
                "message": "; ".join(calc_notes),
                "url": record.get("url_detail"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    if record.get("diagnostics") and isinstance(record["diagnostics"], dict):
        diag = record["diagnostics"]
        if diag.get("detail_status") and diag.get("detail_status") != 200:
            logs.append(
                {
                    "stage": "card",
                    "message": f"HTTP статус {diag.get('detail_status')}",
                    "url": record.get("url_detail") or record.get("url_listing"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

    row = {
        "Источник": "Avito",
        "Площадка ID": record.get("external_id"),
        "Ссылка": record.get("url_detail") or record.get("url_listing"),
        "Город": city,
        "Регион": region,
        "Должность": title,
        "Компания": company,
        "ЗП от (тыс)": salary.salary_from_thousands,
        "ЗП до (тыс)": salary.salary_to_thousands,
        "Ставка в час (руб)": salary.hourly_rate,
        "Средний доход за смену 12ч": salary.shift_income,
        "График": schedule,
        "Формат работы": format_work,
        "Опыт": experience_text,
        "Обязанности": duties,
        "Требования": requirements,
        "Условия/льготы": conditions,
        "Часы работы": working_hours_text,
        "Дата публикации": published.isoformat() if published else "",
        "Дата сбора": collected_at.isoformat(),
        "Контакты": contact_raw,
        "Примечания расчета": "; ".join(calc_notes),
    }

    return row, logs


DATA_COLUMNS = [
    "Источник",
    "Площадка ID",
    "Ссылка",
    "Город",
    "Регион",
    "Должность",
    "Компания",
    "ЗП от (тыс)",
    "ЗП до (тыс)",
    "Ставка в час (руб)",
    "Средний доход за смену 12ч",
    "График",
    "Формат работы",
    "Опыт",
    "Обязанности",
    "Требования",
    "Условия/льготы",
    "Часы работы",
    "Дата публикации",
    "Дата сбора",
    "Контакты",
    "Примечания расчета",
]


def _merge_rows(existing: Dict[str, object], candidate: Dict[str, object]) -> Dict[str, object]:
    if not existing:
        return candidate
    existing_score = sum(1 for key, val in existing.items() if key in DATA_COLUMNS and val not in (None, ""))
    candidate_score = sum(1 for key, val in candidate.items() if key in DATA_COLUMNS and val not in (None, ""))
    return candidate if candidate_score >= existing_score else existing


def normalize_records(
    records: Sequence[Dict[str, object]],
    config: AvitoPipelineConfig,
    collected_at: Optional[date] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    collected_date = collected_at or datetime.now().date()
    dedup: Dict[str, Dict[str, object]] = {}
    logs: List[Dict[str, object]] = []

    for record in records:
        row, row_logs = _normalize_record(record, config, collected_date)
        key = str(row.get("Площадка ID") or row.get("Ссылка"))
        dedup[key] = _merge_rows(dedup.get(key, {}), row)
        logs.extend(row_logs)

    df = pd.DataFrame(list(dedup.values()), columns=DATA_COLUMNS)

    # Apply salary filters after normalisation (values are in thousands of RUB).
    if config.min_salary is not None:
        df = df[df["ЗП от (тыс)"].fillna(df["ЗП до (тыс)"]).ge(config.min_salary)]
    if config.max_salary is not None:
        df = df[df["ЗП до (тыс)"].fillna(df["ЗП от (тыс)"]).le(config.max_salary)]

    # Round numeric metrics to integers as per specification.
    for column in ["ЗП от (тыс)", "ЗП до (тыс)", "Ставка в час (руб)", "Средний доход за смену 12ч"]:
        if column in df.columns:
            df[column] = df[column].astype(float).round(0)
            df[column] = df[column].where(pd.notna(df[column]), None)

    logs_df = pd.DataFrame(logs, columns=["stage", "message", "url", "timestamp"])
    return df, logs_df


# ---------------------------------------------------------------------------
# Analytics


@dataclass
class AnalyticsResult:
    total: int
    jobs: pd.DataFrame
    cities: pd.DataFrame
    benefits: pd.DataFrame
    experience_share: pd.DataFrame
    format_share: pd.DataFrame
    publication_hist: pd.DataFrame
    hourly_share: float


def _aggregate_job_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "Должность",
            "Количество",
            "ЗП от мин",
            "ЗП от медиана",
            "ЗП от средн",
            "ЗП от макс",
            "ЗП до мин",
            "ЗП до медиана",
            "ЗП до средн",
            "ЗП до макс",
            "Ставка в час мин",
            "Ставка в час медиана",
            "Ставка в час средн",
            "Ставка в час макс",
        ])

    rows = []
    for title, group in df.groupby("Должность"):
        entry = {"Должность": title or "(не указано)", "Количество": int(len(group))}
        for column, prefix in [
            ("ЗП от (тыс)", "ЗП от"),
            ("ЗП до (тыс)", "ЗП до"),
            ("Ставка в час (руб)", "Ставка в час"),
        ]:
            values = group[column].dropna().astype(float)
            if values.empty:
                entry[f"{prefix} мин"] = None
                entry[f"{prefix} медиана"] = None
                entry[f"{prefix} средн"] = None
                entry[f"{prefix} макс"] = None
            else:
                entry[f"{prefix} мин"] = round(float(values.min()), 1)
                entry[f"{prefix} медиана"] = round(float(values.median()), 1)
                entry[f"{prefix} средн"] = round(float(values.mean()), 1)
                entry[f"{prefix} макс"] = round(float(values.max()), 1)
        rows.append(entry)
    result = pd.DataFrame(rows)
    return result.sort_values(by="Количество", ascending=False, ignore_index=True)


def _top_cities(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    counts = df["Город"].fillna("(не указано)").value_counts().head(limit)
    return counts.rename_axis("Город").reset_index(name="Количество")


def _top_benefits(df: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for cell in df["Условия/льготы"].dropna():
        for item in _split_benefits(str(cell)):
            if item:
                counter[item] += 1
    if not counter:
        return pd.DataFrame(columns=["Льгота", "Частота"])
    most_common = counter.most_common(limit)
    return pd.DataFrame(most_common, columns=["Льгота", "Частота"])


def _share_series(series: pd.Series) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=["Категория", "Доля %"])
    counts = series.fillna("(не указано)").value_counts(normalize=True) * 100
    return counts.round(1).rename_axis("Категория").reset_index(name="Доля %")


def _publication_hist(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df["Дата публикации"], errors="coerce")
    counts = dates.dropna().dt.date.value_counts().sort_index()
    return counts.rename_axis("Дата").reset_index(name="Количество")


def build_analytics(df: pd.DataFrame) -> AnalyticsResult:
    total = int(len(df))
    jobs = _aggregate_job_stats(df)
    cities = _top_cities(df)
    benefits = _top_benefits(df)
    experience = _share_series(df["Опыт"])
    formats = _share_series(df["Формат работы"])
    hist = _publication_hist(df)
    hourly_share = float(df["Ставка в час (руб)"].notna().mean() * 100.0) if total else 0.0
    return AnalyticsResult(total, jobs, cities, benefits, experience, formats, hist, hourly_share)


# ---------------------------------------------------------------------------
# Excel export


def _write_dataframe(writer: pd.ExcelWriter, df: pd.DataFrame, sheet: str, startrow: int = 0, startcol: int = 0) -> int:
    df.to_excel(writer, sheet_name=sheet, startrow=startrow, startcol=startcol, index=False)
    return startrow + len(df) + 2


def export_avito_to_excel(
    data_df: pd.DataFrame,
    analytics: AnalyticsResult,
    logs_df: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        data_df.to_excel(writer, sheet_name="Данные", index=False)
        logs_df.to_excel(writer, sheet_name="Логи", index=False)

        workbook = writer.book
        sheet = workbook.add_worksheet("Аналитика")
        writer.sheets["Аналитика"] = sheet

        row = 0
        sheet.write(row, 0, "Общее кол-во вакансий")
        sheet.write(row, 1, analytics.total)
        sheet.write(row, 3, "% записей со ставкой в час")
        sheet.write(row, 4, round(analytics.hourly_share, 1))
        row += 2

        sheet.write(row, 0, "По должностям")
        row += 1
        row = _write_dataframe(writer, analytics.jobs, "Аналитика", startrow=row, startcol=0)

        sheet.write(row, 0, "По городам (топ-10)")
        row += 1
        row = _write_dataframe(writer, analytics.cities, "Аналитика", startrow=row, startcol=0)

        sheet.write(row, 0, "Часто встречающиеся льготы (top-15)")
        row += 1
        row = _write_dataframe(writer, analytics.benefits, "Аналитика", startrow=row, startcol=0)

        sheet.write(row, 0, "Доли по типу опыта")
        row += 1
        row = _write_dataframe(writer, analytics.experience_share, "Аналитика", startrow=row, startcol=0)

        sheet.write(row, 0, "Доли по формату работы")
        row += 1
        row = _write_dataframe(writer, analytics.format_share, "Аналитика", startrow=row, startcol=0)

        sheet.write(row, 0, "Кол-во объявлений по дате публикации")
        row += 1
        _write_dataframe(writer, analytics.publication_hist, "Аналитика", startrow=row, startcol=0)

        data_sheet = writer.sheets["Данные"]
        data_sheet.freeze_panes(1, 0)
        data_sheet.autofilter(0, 0, max(len(data_df), 1), len(DATA_COLUMNS) - 1)

        logs_sheet = writer.sheets["Логи"]
        if not logs_df.empty:
            logs_sheet.freeze_panes(1, 0)


# ---------------------------------------------------------------------------
# High level orchestration


def fetch_avito_vacancies(config: AvitoPipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    avito_config = AvitoConfig(
        regions=[config.region or config.city or "all"],
        queries=config.queries or [""],
        categories=[],
        date_range=config.date_from_days,
        max_pages_per_feed=config.pages_limit,
        base_delay=sum(config.delay_range) / 2.0,
        max_retries=3,
        qa_output_dir=Path("Exports/avito_qa"),
    )

    records = collect_avito_vacancies(avito_config)
    data_df, logs_df = normalize_records(records, config)
    return data_df, logs_df


__all__ = [
    "AvitoPipelineConfig",
    "AnalyticsResult",
    "DATA_COLUMNS",
    "DEFAULT_OUTPUT_PATH",
    "build_analytics",
    "export_avito_to_excel",
    "fetch_avito_vacancies",
    "load_config",
    "normalize_records",
]
