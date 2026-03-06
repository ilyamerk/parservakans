import argparse
import dataclasses
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

EXPORT_DIR = Path("Exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

HH_API = "https://api.hh.ru/vacancies"

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class HHHTTPConfig:
    connect_timeout: float = 5.0
    read_timeout: float = 15.0
    max_retries: int = 4
    backoff_factor: float = 0.6
    pause_between_requests: float = 0.2


DEFAULT_HTTP_CONFIG = HHHTTPConfig()
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclasses.dataclass
class WorkingHoursContext:
    text: str
    url: str = ""
    selector: str = ""
    weight: int = 1


@dataclasses.dataclass
class ShiftLength:
    hours: Optional[float]
    source: str = "unresolved"
    confidence: str = "low"
    notes: list[str] = dataclasses.field(default_factory=list)
    ambiguous: bool = False


def _get_sess() -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": "parservakans/1.0"})
    adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=Retry(total=0, redirect=0))
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def _request_json_with_retries(
    sess: requests.Session,
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    config: HHHTTPConfig = DEFAULT_HTTP_CONFIG,
    stage: str,
) -> tuple[Optional[dict[str, Any]], int, Optional[str]]:
    last_error: Optional[str] = None
    for attempt in range(1, config.max_retries + 1):
        try:
            resp: Response = sess.get(url, params=params, timeout=(config.connect_timeout, config.read_timeout))
            if resp.status_code in RETRYABLE_STATUS_CODES:
                last_error = f"HTTP {resp.status_code}"
                logger.warning(
                    "HH API retriable response: stage=%s attempt=%s/%s url=%s params=%s reason=%s",
                    stage,
                    attempt,
                    config.max_retries,
                    url,
                    params,
                    last_error,
                )
            elif resp.status_code >= 400:
                last_error = f"HTTP {resp.status_code}"
                logger.error(
                    "HH API non-retriable response: stage=%s url=%s params=%s reason=%s",
                    stage,
                    url,
                    params,
                    last_error,
                )
                return None, attempt, last_error
            else:
                return resp.json(), attempt, None
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "HH API timeout: stage=%s attempt=%s/%s url=%s params=%s reason=%s",
                stage,
                attempt,
                config.max_retries,
                url,
                params,
                last_error,
            )
        except requests.exceptions.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "HH API request error: stage=%s attempt=%s/%s url=%s params=%s reason=%s",
                stage,
                attempt,
                config.max_retries,
                url,
                params,
                last_error,
            )

        if attempt < config.max_retries:
            sleep_s = config.backoff_factor * (2 ** (attempt - 1))
            time.sleep(sleep_s)

    logger.error(
        "HH API final failure: stage=%s attempts=%s url=%s params=%s final_reason=%s",
        stage,
        config.max_retries,
        url,
        params,
        last_error,
    )
    return None, config.max_retries, last_error


def hh_search(
    query: str,
    area: int = 1,
    pages: int = 3,
    per_page: int = 50,
    pause: float = 0.2,
    search_in: str = "name",
    start_page: int = 0,
    timeout: float = 15,
    config: Optional[HHHTTPConfig] = None,
) -> list[dict[str, Any]]:
    http_config = config or HHHTTPConfig(read_timeout=timeout, pause_between_requests=pause)
    sess = _get_sess()
    items: list[dict[str, Any]] = []
    page_errors: list[dict[str, Any]] = []
    for page in range(start_page, start_page + pages):
        params = {
            "text": query,
            "area": area,
            "page": page,
            "per_page": per_page,
            "search_field": search_in,
        }
        payload, attempts, error_reason = _request_json_with_retries(
            sess,
            HH_API,
            params=params,
            config=http_config,
            stage="поиск вакансий",
        )
        if payload is None:
            page_errors.append({"page": page, "attempts": attempts, "reason": error_reason})
            logger.error(
                "Не удалось загрузить страницу поиска HH, страница пропущена: page=%s attempts=%s reason=%s",
                page,
                attempts,
                error_reason,
            )
            continue

        page_items = payload.get("items", []) or []
        items.extend(page_items)
        if page >= int(payload.get("pages", 1)) - 1:
            break

        if http_config.pause_between_requests:
            time.sleep(http_config.pause_between_requests)

    if not items and page_errors:
        raise RuntimeError(
            "HH API временно недоступен: не удалось загрузить страницы поиска. "
            "Попробуйте позже или уменьшите нагрузку (pages/per_page/workers)."
        )
    return items


def load_hh_vacancy_details(
    vacancy_id: str,
    *,
    sess: Optional[requests.Session] = None,
    config: HHHTTPConfig = DEFAULT_HTTP_CONFIG,
) -> Optional[dict[str, Any]]:
    session = sess or _get_sess()
    payload, _, _ = _request_json_with_retries(
        session,
        f"{HH_API}/{vacancy_id}",
        config=config,
        stage="загрузка детальной вакансии",
    )
    return payload


def is_hourly_rate(text: str) -> bool:
    t = (text or "").lower()
    return bool(re.search(r"(/\s*час|руб\s*/\s*ч|per\s*hour|/hour)", t))


def is_shift_rate(text: str) -> bool:
    t = (text or "").lower()
    return bool(re.search(r"(за\s*смен|/\s*смен|per\s*shift|смена)\b", t))


def extract_rate(text: str) -> Optional[dict[str, Any]]:
    t = (text or "").replace("−", "-").replace("–", "-")
    currency = None
    if "€" in t:
        currency = "EUR"
    elif "$" in t:
        currency = "USD"
    elif re.search(r"(руб|₽)", t.lower()):
        currency = "RUB"

    nums = [float(x.replace(" ", "").replace(",", ".")) for x in re.findall(r"\d[\d\s]*(?:[\.,]\d+)?", t)]
    if not nums:
        return None
    lo, hi = (nums[0], nums[0]) if len(nums) == 1 else (min(nums[0], nums[1]), max(nums[0], nums[1]))

    if re.search(r"от\s*\d", t.lower()) and re.search(r"до\s*\d", t.lower()):
        value = re.search(r"(от\s*[\d\s]+\s*до\s*[\d\s]+)", t.lower())
        rendered = (value.group(1) if value else f"{int(lo)}–{int(hi)}").strip()
    elif len(nums) > 1:
        prefix = "€" if currency == "EUR" else "$" if currency == "USD" else ""
        suffix = " ₽" if currency == "RUB" else ""
        rendered = f"{prefix}{int(lo)}–{int(hi)}{suffix}".strip()
    else:
        prefix = "€" if currency == "EUR" else "$" if currency == "USD" else ""
        suffix = " ₽" if currency == "RUB" else ""
        rendered = f"{prefix}{int(lo)}{suffix}".strip()
        if "+" in t:
            rendered += " + %"

    return {"value": rendered, "min": lo, "max": hi, "currency": currency}


def collect_rate_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        text = str(row.get("__rate_text") or "")
        url = str(row.get("Ссылка") or "")
        for m in re.finditer(r"(\d+[\d\s]*)\s*(?:/\s*час|руб\s*/\s*ч)", text.lower()):
            out.append({"type": "hourly", "value": m.group(1).strip(), "url": url})
        for m in re.finditer(r"(\d+[\d\s]*)\s*(?:/\s*смен|за\s*смен)", text.lower()):
            out.append({"type": "shift", "value": m.group(1).strip(), "url": url})
    return out


def filter_rows_without_shift_rates(rows: list[dict[str, Any]], rate_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    shift_urls = {r["url"] for r in rate_rows if r.get("type") == "shift"}
    return [r for r in rows if r.get("Ссылка") not in shift_urls]


def is_gross_salary(text: str) -> bool:
    t = (text or "").lower()
    if re.search(r"(на\s*руки|\bnet\b)", t):
        return False
    return bool(re.search(r"(до\s*вычета\s*ндфл|\bgross\b|брутто)", t))


def _parse_hour_value(raw: str) -> Optional[float]:
    if not raw:
        return None
    return float(raw.replace(",", "."))


def extract_shift_len(text: str) -> ShiftLength:
    t = (text or "").lower()
    candidates: list[float] = []

    for m in re.finditer(r"(?:смен[аы]\s*по|по)\s*(\d+(?:[\.,]\d+)?)\s*час", t):
        v = _parse_hour_value(m.group(1))
        if v:
            candidates.append(v)

    for m in re.finditer(r"(?:смена\s*)?(\d+(?:[\.,]\d+)?)(?:-?часовая|\s*час(?:ов|а)?)", t):
        ctx = t[max(0, m.start()-12):m.start()]
        if "перерыв" in ctx and "смен" not in ctx:
            continue
        v = _parse_hour_value(m.group(1))
        if v:
            candidates.append(v)

    for tr in re.finditer(r"(?:с\s*)?(\d{1,2}):(\d{2})\s*(?:до|\-|–|—)\s*(\d{1,2}):(\d{2})", t):
        h1 = int(tr.group(1)) + int(tr.group(2)) / 60
        h2 = int(tr.group(3)) + int(tr.group(4)) / 60
        dur = h2 - h1
        if dur <= 0:
            dur += 24
        candidates.append(round(dur, 2))

    if not candidates:
        return ShiftLength(hours=None, source="unresolved", confidence="low", notes=["shift duration not found"])

    unique = sorted(set(candidates))
    return ShiftLength(
        hours=unique[0],
        source="description_explicit_hours" if len(unique) == 1 else "description_multiple_variants",
        confidence="high",
        notes=[] if len(unique) == 1 else ["multiple variants detected"],
        ambiguous=len(unique) > 1,
    )




def extract_employment_type(text: str, employment_name: str = "") -> tuple[Optional[str], list[str]]:
    combined = " ".join(filter(None, [text, employment_name])).lower()
    notes: list[str] = []
    tk = bool(
        re.search(
            r"(по\s*тк\s*рф|по\s*тк\b|трудов(?:ой|ого)?\s*договор|официальн\w*\s*(?:оформлен|трудоустрой))",
            combined,
        )
    )
    gph = bool(
        re.search(
            r"(\bгпх\b|договор\s*гпх|гражданско-?правов(?:ой|ого)?\s*договор|самозанят|подряд)",
            combined,
        )
    )
    if tk and gph:
        notes.append("both ТК and ГПХ found")
        return "ТК / ГПХ", notes
    if tk:
        return "ТК", notes
    if gph:
        return "ГПХ", notes
    return None, ["employment type unresolved"]


def extract_schedule(text: str, schedule_name: str = "") -> tuple[Optional[str], str]:
    combined = " ".join(filter(None, [text, schedule_name])).lower()
    m = re.search(r"\b(\d{1,2})\s*/\s*(\d{1,2})\b", combined)
    if m:
        return f"{int(m.group(1))}/{int(m.group(2))}", "description_or_structured:pattern"
    if "сменн" in combined:
        return "сменный", "description_or_structured:keyword"
    if "гибк" in combined:
        return "гибкий", "description_or_structured:keyword"
    return None, "unresolved"


def extract_payment_frequency(text: str) -> tuple[Optional[str], str]:
    t = (text or "").lower()
    patterns = [
        (r"(ежедневн\w*\s*выплат|выплат\w*\s*ежедневн)", "ежедневно"),
        (r"(еженедельн\w*\s*выплат|раз\s*в\s*недел|выплат\w*\s*раз\s*в\s*недел)", "еженедельно"),
        (r"(2\s*раза\s*в\s*месяц|дважды\s*в\s*месяц)", "2 раза в месяц"),
        (r"(ежемесячн\w*|раз\s*в\s*месяц)", "ежемесячно"),
        (r"(после\s*смен|в\s*конце\s*смен)", "после смены"),
    ]
    for pattern, value in patterns:
        if re.search(pattern, t):
            return value, "description_or_structured"
    return None, "unresolved"


def extract_benefits(text: str) -> list[str]:
    t = (text or "").lower()
    benefit_patterns = [
        (r"\bдмс\b|добровольн\w*\s*медицинск\w*\s*страх", "ДМС"),
        (r"бесплатн\w*\s*питан|питан\w*\s*за\s*сч[её]т\s*компан|компенсац\w*\s*питан", "питание"),
        (r"скидк\w*\s*сотрудник|корпоративн\w*\s*скидк", "скидки сотрудникам"),
        (r"оплачиваем\w*\s*обучен|обучен\w*\s*за\s*сч[её]т\s*компан", "обучение"),
        (r"\bуниформ\w*\b|\bформ\w*\b", "униформа"),
        (r"развозк\w*|корпоративн\w*\s*транспорт|компенсац\w*\s*проезд", "транспорт"),
        (r"бонус\w*|преми\w*", "бонусы/премии"),
        (r"медкнижк\w*\s*за\s*сч[её]т\s*работодат", "медкнижка за счёт работодателя"),
        (r"проживан\w*|жиль[её]", "проживание"),
        (r"подарк\w*\s*дет|подарк\w*\s*сотрудник", "подарки"),
        (r"спорт|фитнес", "спорт/фитнес"),
        (r"страхован", "страхование"),
        (r"карьерн\w*\s*рост", "карьерный рост"),
    ]
    found: list[str] = []
    for pattern, normalized in benefit_patterns:
        if re.search(pattern, t):
            found.append(normalized)
    return list(dict.fromkeys(found))


def extract_shift_payment(text: str) -> Optional[float]:
    t = (text or "").lower().replace("\xa0", " ")
    m = re.search(r"(\d[\d\s]*(?:[\.,]\d+)?)\s*(?:руб(?:\.|лей|ля)?|₽)?\s*(?:за\s*смен|/\s*смен)", t)
    if not m:
        return None
    return float(m.group(1).replace(" ", "").replace(",", "."))


def extract_hourly_payment(text: str) -> Optional[float]:
    t = (text or "").lower().replace("\xa0", " ")
    m = re.search(r"(\d[\d\s]*(?:[\.,]\d+)?)\s*(?:руб(?:\.|лей|ля)?|₽)?\s*(?:/\s*час|в\s*час|за\s*час)", t)
    if not m:
        return None
    return float(m.group(1).replace(" ", "").replace(",", "."))

SCHEDULE_SHIFTS_PER_MONTH = {
    "2/2": 15.0,
    "5/2": 22.0,
    "3/3": 15.0,
    "6/1": 26.0,
    "7/7": 14.0,
    "4/3": 17.0,
    "4/4": 13.0,
}


def compute_hourly_rate(
    hourly_rate: Optional[float],
    shift_rate: Optional[float],
    shift_info: Optional[ShiftLength],
    monthly_salary: Optional[float] = None,
    schedule: Optional[str] = None,
):
    if hourly_rate:
        return round(float(hourly_rate), 2), "provided:hourly", []
    if shift_rate and shift_info and shift_info.hours and not shift_info.ambiguous:
        return round(float(shift_rate) / float(shift_info.hours), 2), "exact:shift_rate/shift_duration", []
    if shift_rate and shift_info and shift_info.ambiguous:
        return None, "unresolved:ambiguous_shift_duration", ["multiple shift durations"]
    shifts_per_month = SCHEDULE_SHIFTS_PER_MONTH.get(str(schedule or ""))
    if monthly_salary and shift_info and shift_info.hours and not shift_info.ambiguous and shifts_per_month:
        hourly = float(monthly_salary) / (float(shift_info.hours) * shifts_per_month)
        return round(hourly, 2), "calculated:monthly_salary/(shift_duration*shifts_per_month)", []
    if monthly_salary and not shifts_per_month:
        return None, "unresolved:unknown_schedule_for_monthly_salary", ["monthly salary provided but schedule not supported"]
    return None, "unresolved:insufficient_data", ["missing hourly and shift duration"]




def compute_shift_income_total(hourly_rate: Optional[float], shift_info: Optional[ShiftLength]):
    if hourly_rate is None:
        return None, "unresolved:missing_hourly_rate"
    if not shift_info or shift_info.hours is None:
        return None, "unresolved:missing_shift_duration"
    if shift_info.ambiguous:
        return None, "unresolved:ambiguous_shift_duration"
    return round(float(hourly_rate) * float(shift_info.hours), 2), "exact:hourly_rate*shift_duration"


_DAY_MAP = {"пн": "Mon", "вт": "Tue", "ср": "Wed", "чт": "Thu", "пт": "Fri", "сб": "Sat", "вс": "Sun", "mon": "Mon", "tue": "Tue", "wed": "Wed", "thu": "Thu", "fri": "Fri", "sat": "Sat", "sun": "Sun"}


def extract_working_hours(contexts: list[WorkingHoursContext]) -> dict[str, Any]:
    raw_matches = [c.text for c in contexts if c.text]
    text = "\n".join(raw_matches)
    text_l = text.lower()
    by_days: list[dict[str, str]] = []
    schedule_hint = None
    is_247 = bool(re.search(r"(круглосуточно|24/7)", text_l))
    lang = "en" if re.search(r"\bmon|opening hours\b", text_l) else "ru"
    notes = []

    if re.search(r"\b2\s*/\s*2\b", text_l):
        schedule_hint = "2/2"
    sh = re.search(r"\b(\d/\d)\b", text_l)
    shift_len = re.search(r"(\d+(?:[\.,]\d+)?)\s*час", text_l)

    # simple day-range parser for tests
    m = re.search(r"(пн|mon)\s*[–-]\s*(пт|fri)\s*(\d{1,2})(?::(\d{2}))?(am|pm)?\s*[–-]\s*(\d{1,2})(?::(\d{2}))?(am|pm)?", text_l)
    if m:
        s_h, s_m = int(m.group(3)), int(m.group(4) or 0)
        if m.group(5) == "pm" and s_h < 12:
            s_h += 12
        e_h, e_m = int(m.group(6)), int(m.group(7) or 0)
        if m.group(8) == "pm" and e_h < 12:
            e_h += 12
        for day in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
            by_days.append({"day": day, "start": f"{s_h:02d}:{s_m:02d}", "end": f"{e_h:02d}:{e_m:02d}"})

    m_all = re.search(r"пн\s*[–-]\s*вс\s*(\d{1,2}):(\d{2})\s*[–-]\s*(\d{1,2}):(\d{2})", text_l)
    if m_all:
        s_h, s_m = int(m_all.group(1)), int(m_all.group(2))
        e_h, e_m = int(m_all.group(3)), int(m_all.group(4))
        by_days = [{"day": d, "start": f"{s_h:02d}:{s_m:02d}", "end": f"{e_h:02d}:{e_m:02d}"} for d in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]]

    sat = re.search(r"(сб|sat)\s*(\d{1,2})(?::(\d{2}))?(?:am|pm)?\s*[–-]\s*(\d{1,2})(?::(\d{2}))?(am|pm)?", text_l)
    if sat:
        shh = int(sat.group(2)); smm = int(sat.group(3) or 0)
        ehh = int(sat.group(4)); emm = int(sat.group(5) or 0)
        if sat.group(6) == "pm" and ehh < 12:
            ehh += 12
        by_days.append({"day": "Sat", "start": f"{shh:02d}:{smm:02d}", "end": f"{ehh:02d}:{emm:02d}"})

    if "вс выход" in text_l:
        notes.append("Sun is off")

    confidence = 0.3
    if by_days:
        confidence = 0.95
    elif schedule_hint:
        confidence = 0.6
    elif is_247:
        confidence = 0.95

    return {
        "raw_matches": raw_matches,
        "normalized": {
            "by_days": by_days,
            "shift_based": {
                "pattern": sh.group(1) if sh else schedule_hint,
                "shift_length_hours": float(shift_len.group(1).replace(",", ".")) if shift_len else None,
            },
            "is_247": is_247,
        },
        "schedule_hint": schedule_hint,
        "confidence": confidence,
        "lang": lang,
        "notes": "; ".join(notes),
        "source": [{"selector": c.selector, "url": c.url, "weight": c.weight} for c in contexts],
        "log": ["parsed"],
    }


def map_hh(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for it in items:
        salary = it.get("salary") or {}
        schedule_name = str((it.get("schedule") or {}).get("name", "") or "")
        employment_name = str((it.get("employment") or {}).get("name", "") or "")
        desc_text = " ".join(
            filter(
                None,
                [
                    str(it.get("name", "") or ""),
                    str((it.get("snippet") or {}).get("requirement", "") or ""),
                    str((it.get("snippet") or {}).get("responsibility", "") or ""),
                    schedule_name,
                    employment_name,
                ],
            )
        )

        sl = extract_shift_len(desc_text)
        employment_type, employment_notes = extract_employment_type(desc_text, employment_name)
        schedule, schedule_source = extract_schedule(desc_text, schedule_name)
        hourly_from_text = extract_hourly_payment(desc_text)
        shift_rate = extract_shift_payment(desc_text)
        monthly_salary = salary.get("from") or salary.get("to")
        hr, hr_method, hr_notes = compute_hourly_rate(
            hourly_from_text,
            shift_rate,
            sl,
            monthly_salary=monthly_salary,
            schedule=schedule,
        )
        payment_frequency, payment_frequency_source = extract_payment_frequency(desc_text)
        benefits = extract_benefits(desc_text)
        shift_income_total, shift_income_method = compute_shift_income_total(hr, sl)

        parsing_notes: list[str] = []
        parsing_notes.extend(sl.notes)
        parsing_notes.extend(employment_notes)
        parsing_notes.extend(hr_notes)
        if sl.source == "unresolved":
            parsing_notes.append("shift duration unresolved")
        if employment_type is None:
            parsing_notes.append("employment type unresolved")
        if schedule_source != "unresolved":
            parsing_notes.append(f"schedule source: {schedule_source}")
        else:
            parsing_notes.append("schedule unresolved")
        if payment_frequency_source != "unresolved":
            parsing_notes.append(f"payment frequency source: {payment_frequency_source}")
        else:
            parsing_notes.append("payment frequency unresolved")
        if shift_income_method.startswith("unresolved"):
            parsing_notes.append(shift_income_method)

        rows.append(
            {
                "Должность": it.get("name", ""),
                "Работодатель": (it.get("employer") or {}).get("name", ""),
                "Дата публикации": it.get("published_at", ""),
                "ЗП от (т.р.)": (salary.get("from") or 0) / 1000 if salary.get("from") else None,
                "ЗП до (т.р.)": (salary.get("to") or 0) / 1000 if salary.get("to") else None,
                "Средний совокупный доход при графике 2/2 по 12 часов": shift_income_total,
                "В час": hr,
                "Длительность смены": sl.hours,
                "Требуемый\nопыт": (it.get("experience") or {}).get("name", ""),
                "Труд-во": employment_type,
                "График": schedule,
                "Частота выплат": payment_frequency,
                "Льготы": ", ".join(benefits),
                "Обязаности": str((it.get("snippet") or {}).get("responsibility", "") or ""),
                "Ссылка": (it.get("alternate_url") or ""),
                "Примечание": "; ".join(dict.fromkeys([n for n in parsing_notes if n])),
                "shift_duration_hours": sl.hours,
                "employment_type": employment_type,
                "schedule": schedule,
                "hourly_rate": hr,
                "hourly_rate_method": hr_method,
                "shift_income_total": shift_income_total,
                "parsing_notes": "; ".join(dict.fromkeys([n for n in parsing_notes if n])),
            }
        )
    return rows



def _resolve_export_paths(target: str) -> tuple[Path, Path]:
    path = Path(target)
    if path.suffix.lower() == ".csv":
        return path, path.with_name(path.stem + "_view.xlsx")
    if path.suffix.lower() == ".xlsx":
        return path.with_suffix(".csv"), path
    raise ValueError("Target must be .csv or .xlsx")


def main() -> None:
    ap = argparse.ArgumentParser(description="Парсер вакансий hh.ru")
    ap.add_argument("--query", default="Бариста")
    ap.add_argument("--area", type=int, default=1)
    ap.add_argument("--city", default="Москва")
    ap.add_argument("--pages", type=int, default=3)
    ap.add_argument("--per_page", type=int, default=50)
    ap.add_argument("--pause", type=float, default=0.2)
    ap.add_argument("--search_in", default="name")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--role", default="")
    ap.add_argument("--no_filter", action="store_true")
    ap.add_argument("--out_csv", default=str(EXPORT_DIR / "raw.csv"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        hh_items = hh_search(
            args.query,
            args.area,
            args.pages,
            args.per_page,
            args.pause,
            args.search_in,
            timeout=args.timeout,
            config=HHHTTPConfig(
                connect_timeout=min(args.timeout, 6.0),
                read_timeout=max(args.timeout, 8.0),
                max_retries=DEFAULT_HTTP_CONFIG.max_retries,
                backoff_factor=DEFAULT_HTTP_CONFIG.backoff_factor,
                pause_between_requests=args.pause,
            ),
        )
    except RuntimeError as exc:
        print(f"Ошибка: {exc}")
        return

    rows = map_hh(hh_items)

    if args.role and not args.no_filter:
        role = args.role.lower().strip()
        rows = [r for r in rows if role in str(r.get("Должность", "")).lower()]

    out_csv, out_xlsx = _resolve_export_paths(args.out_csv)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    pd.DataFrame(rows).to_excel(out_xlsx, index=False)
    print(f"HH: {len(rows)} | Total: {len(rows)}")
    print(f"Saved CSV: {out_csv}")
    print(f"Saved XLSX: {out_xlsx}")


if __name__ == "__main__":
    main()
