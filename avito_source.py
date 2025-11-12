"""Avito vacancy collection with polite crawling and rich normalization.

This module implements an Avito client that can be plugged into the existing
vacancy pipeline.  The collector follows the requirements described in the
specification: it navigates Avito listing pages, handles polite networking
behaviour, parses listing cards and detailed pages, performs normalization of
fields (salary, schedule, experience, working hours) and keeps an audit trail
in diagnostics.  It also stores QA samples that can be persisted for manual
inspection.

The implementation is designed to work both in production (using real HTTP
requests through :mod:`requests`) and in tests, where a custom fetcher can be
injected to supply deterministic HTML fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import random
import re
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode, urljoin, urlparse

import requests
from bs4 import BeautifulSoup


LOGGER = logging.getLogger(__name__)


USER_AGENTS = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.5 Safari/605.1.15"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
]


_REGION_SLUGS = {
    "москва": "moskva",
    "moscow": "moskva",
    "санкт-петербург": "sankt-peterburg",
    "санкт петербург": "sankt-peterburg",
    "спб": "sankt-peterburg",
    "ekaterinburg": "ekaterinburg",
    "екатеринбург": "ekaterinburg",
    "новосибирск": "novosibirsk",
    "казань": "kazan",
    "нижний новгород": "nizhniy_novgorod",
    "самара": "samara",
    "ростов-на-дону": "rostov-na-donu",
    "ростов на дону": "rostov-na-donu",
    "уфа": "ufa",
    "красноярск": "krasnoyarsk",
    "пермь": "perm",
    "воронеж": "voronezh",
    "волгоград": "volgograd",
}


def _slugify_region(raw: str) -> str:
    value = (raw or "").strip().lower()
    if not value:
        return "all"
    if value in _REGION_SLUGS:
        return _REGION_SLUGS[value]
    # replace spaces and consecutive dashes with underscores
    value = re.sub(r"[^0-9a-z]+", "_", value, flags=re.IGNORECASE)
    value = value.strip("_")
    return value or "all"


def _slugify_category(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    value = re.sub(r"[^0-9a-z]+", "_", raw.strip().lower())
    value = value.strip("_")
    return value or None


def _normalize_currency(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    val = token.strip().lower()
    if val in {"rub", "р", "руб", "руб.", "₽"}:
        return "RUB"
    if val in {"eur", "€"}:
        return "EUR"
    if val in {"usd", "$"}:
        return "USD"
    if val in {"kzt", "₸"}:
        return "KZT"
    if val in {"uah", "грн", "₴"}:
        return "UAH"
    if val in {"byn", "br"}:
        return "BYN"
    if val in {"gbp", "£"}:
        return "GBP"
    return token.strip().upper()


def _parse_date_range(value: Optional[str | int | float]) -> Optional[timedelta]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value <= 0:
            return None
        return timedelta(days=float(value))
    match = re.search(r"(\d+)", str(value))
    if not match:
        return None
    days = int(match.group(1))
    if days <= 0:
        return None
    return timedelta(days=days)


def _safe_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


class _HostRateLimiter:
    """Simple per-host rate limiter with jitter."""

    def __init__(self, min_delay: float) -> None:
        self._min_delay = max(0.1, float(min_delay))
        self._lock = threading.Lock()
        self._last_call: Dict[str, float] = {}

    def wait(self, host: str) -> None:
        with self._lock:
            now = time.monotonic()
            last = self._last_call.get(host, 0.0)
            delta = now - last
            gap = self._min_delay
            if delta < gap:
                to_sleep = gap - delta + random.uniform(0.05, 0.2)
                time.sleep(max(0.05, to_sleep))
            self._last_call[host] = time.monotonic()


@dataclass
class AvitoConfig:
    regions: List[str]
    queries: List[str]
    categories: List[str] = field(default_factory=list)
    date_range: Optional[str | int | float] = None
    max_pages_per_feed: int = 3
    only_with_salary: bool = False
    request_timeout: float = 15.0
    base_delay: float = 3.0
    max_retries: int = 3
    timezone: timezone = timezone(timedelta(hours=3))  # Europe/Moscow
    qa_output_dir: Optional[Path] = Path("Exports/avito_qa")

    def __post_init__(self) -> None:
        if not self.regions:
            self.regions = ["all"]
        if not self.queries:
            self.queries = [""]
        if self.categories is None:
            self.categories = []
        self.max_pages_per_feed = max(1, int(self.max_pages_per_feed))
        self.base_delay = max(0.2, float(self.base_delay))
        self.request_timeout = max(5.0, float(self.request_timeout))
        self.max_retries = max(1, int(self.max_retries))
        if isinstance(self.qa_output_dir, str):
            self.qa_output_dir = Path(self.qa_output_dir)

    @property
    def min_allowed_date(self) -> Optional[datetime]:
        delta = _parse_date_range(self.date_range)
        if not delta:
            return None
        now = datetime.now(self.timezone)
        return now - delta


@dataclass
class ListingCard:
    external_id: int
    url_listing: str
    url_detail: str
    title: str
    salary_text: str
    location_text: str
    address_raw: Optional[str]
    posted_at_raw: Optional[str]
    is_promoted: bool
    is_featured: bool
    snippet_text: Optional[str]
    raw_html: str
    diagnostics: Dict[str, List[str]] = field(default_factory=dict)


def _extract_external_id(url: str) -> Optional[int]:
    match = re.search(r"-(\d{6,})", url)
    if not match:
        return None
    return int(match.group(1))


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _compute_hash(text: str) -> str:
    data = text.encode("utf-8", errors="ignore")
    return hashlib.sha256(data).hexdigest()


def _parse_relative_datetime(raw: str, now: datetime) -> Optional[datetime]:
    if not raw:
        return None
    text = raw.strip().lower()
    if not text:
        return None

    def _combine(date_part: datetime, time_part: Optional[Tuple[int, int]]) -> datetime:
        if not time_part:
            return date_part.replace(hour=0, minute=0, second=0, microsecond=0)
        hour, minute = time_part
        return date_part.replace(hour=hour, minute=minute, second=0, microsecond=0)

    def _extract_time(segment: str) -> Optional[Tuple[int, int]]:
        m = re.search(r"(\d{1,2}):(\d{2})", segment)
        if not m:
            return None
        h, m_ = int(m.group(1)), int(m.group(2))
        if 0 <= h < 24 and 0 <= m_ < 60:
            return h, m_
        return None

    time_part = _extract_time(text)
    if "сегодня" in text:
        return _combine(now, time_part)
    if "вчера" in text:
        return _combine(now - timedelta(days=1), time_part)

    hour_match = re.search(r"(\d+)\s*(?:час|ч)", text)
    if "назад" in text and hour_match:
        hours = int(hour_match.group(1))
        if hours >= 0:
            return now - timedelta(hours=hours)

    day_match = re.search(r"(\d+)\s*(?:дн|day)", text)
    if "назад" in text and day_match:
        days = int(day_match.group(1))
        if days >= 0:
            base = now - timedelta(days=days)
            return _combine(base, time_part)

    week_match = re.search(r"(\d+)\s*(?:нед|week)", text)
    if "назад" in text and week_match:
        weeks = int(week_match.group(1))
        if weeks >= 0:
            base = now - timedelta(weeks=weeks)
            return _combine(base, time_part)

    # absolute date like "12 мая" or "12.05.2024"
    m = re.search(r"(\d{1,2})\.(\d{1,2})(?:\.(\d{2,4}))?", text)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
        year = int(m.group(3)) if m.group(3) else now.year
        try:
            dt = now.replace(year=year, month=month, day=day)
            return _combine(dt, time_part)
        except ValueError:
            return None

    return None


def _detect_salary_period(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.lower()
    if re.search(r"в\s*час|/\s*ч|за\s*час|per\s*hour|hourly", t):
        return "per_hour"
    if re.search(r"за\s*смен", t) or re.search(r"per\s*shift", t):
        return "per_shift"
    if re.search(r"в\s*день|per\s*day", t):
        return "per_day"
    if re.search(r"в\s*недел", t):
        return "per_week"
    if re.search(r"в\s*месяц|per\s*month", t):
        return "per_month"
    return None


_RATE_NUMBER_RE = re.compile(r"\d[\d\s\u202f]*(?:[\.,]\d+)?")
_CURRENCY_RE = re.compile(
    r"₽|руб\.?|rur|rub|uah|грн|₴|eur|€|usd|\$|kzt|₸|byn|br|gbp|£",
    re.IGNORECASE,
)


def _extract_salary_numbers(text: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    segment = _clean_text(text)
    if not segment:
        return None, None, None
    matches = list(_RATE_NUMBER_RE.finditer(segment))
    if not matches:
        return None, None, None
    numbers: List[float] = []
    for match in matches[:2]:
        cleaned = match.group(0).replace("\u202f", "").replace(" ", "").replace(",", ".")
        try:
            numbers.append(float(cleaned))
        except ValueError:
            continue
    if not numbers:
        return None, None, None
    minimum = numbers[0]
    maximum = numbers[-1] if len(numbers) > 1 else numbers[0]
    if minimum > maximum:
        minimum, maximum = maximum, minimum
    currency_match = _CURRENCY_RE.search(segment)
    currency = _normalize_currency(currency_match.group(0)) if currency_match else None
    return minimum, maximum, currency


def _extract_section_text(soup: BeautifulSoup, keywords: Iterable[str]) -> Optional[str]:
    if not soup:
        return None
    patterns = [re.compile(k, re.IGNORECASE) for k in keywords]
    for header in soup.find_all(["h2", "h3", "strong"]):
        text = header.get_text(strip=True)
        if not text:
            continue
        if any(p.search(text) for p in patterns):
            parts: List[str] = []
            for sibling in header.next_siblings:
                if getattr(sibling, "name", None) in {"h2", "h3", "strong"}:
                    break
                if isinstance(sibling, str):
                    val = sibling.strip()
                    if val:
                        parts.append(val)
                    continue
                if getattr(sibling, "name", "").lower() in {"ul", "ol"}:
                    for li in sibling.find_all("li"):
                        li_text = li.get_text(" ", strip=True)
                        if li_text:
                            parts.append(li_text)
                    continue
                val = sibling.get_text(" ", strip=True)
                if val:
                    parts.append(val)
            return "\n".join(parts).strip() or None
    return None


def _gather_benefits(text: str) -> List[str]:
    if not text:
        return []
    benefits = []
    for line in text.splitlines():
        cleaned = line.strip(" -•\t")
        if not cleaned:
            continue
        if re.search(r"(питани|обеды|форма|медкниж|страхов|проезд|сменн\w* график)", cleaned, re.IGNORECASE):
            benefits.append(cleaned)
    return benefits


def _normalize_schedule_hint(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"(\d+\/\d+)", text)
    if m:
        return m.group(1)
    if re.search(r"вахт", text, re.IGNORECASE):
        return "вахта"
    if re.search(r"гибк", text, re.IGNORECASE):
        return "гибкий график"
    return None


class AvitoCollector:
    """Collect vacancies from Avito with detailed parsing."""

    def __init__(
        self,
        config: AvitoConfig,
        fetcher: Optional[Callable[[str], requests.Response]] = None,
        now: Optional[datetime] = None,
    ) -> None:
        self.config = config
        self.now = now or datetime.now(config.timezone)
        self._limiter = _HostRateLimiter(config.base_delay)
        self._fetcher = fetcher or self._http_fetch
        self._session: Optional[requests.Session] = None
        self._ua_index = 0
        self.records: Dict[int, Dict] = {}
        self.stats = {
            "listing_cards": 0,
            "detail_success": 0,
            "detail_fail": 0,
            "blocked": 0,
        }
        self.qa_samples: List[Dict[str, object]] = []

    # ------------------------------------------------------------------
    # Networking helpers
    def _session_with_retries(self) -> requests.Session:
        if self._session is None:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(max_retries=self.config.max_retries)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            session.headers.update({"User-Agent": USER_AGENTS[self._ua_index]})
            self._session = session
        return self._session

    def _rotate_user_agent(self) -> None:
        self._ua_index = (self._ua_index + 1) % len(USER_AGENTS)
        if self._session is not None:
            self._session.headers.update({"User-Agent": USER_AGENTS[self._ua_index]})

    def _http_fetch(self, url: str) -> requests.Response:
        session = self._session_with_retries()
        parsed = urlparse(url)
        self._limiter.wait(parsed.netloc)
        delay = self.config.base_delay
        for attempt in range(self.config.max_retries):
            try:
                response = session.get(url, timeout=self.config.request_timeout)
            except requests.RequestException as exc:  # pragma: no cover - network errors
                LOGGER.warning("Avito fetch failed: %s", exc)
                time.sleep(delay)
                delay *= 2
                continue

            if response.status_code in {429, 403}:
                self.stats["blocked"] += 1
                self._rotate_user_agent()
                time.sleep(delay + random.uniform(0.5, 1.0))
                delay *= 2
                continue
            if response.status_code >= 500:
                time.sleep(delay)
                delay *= 2
                continue
            return response
        return response

    # ------------------------------------------------------------------
    # Parsing
    def _build_listing_url(self, region: str, category: Optional[str], query: str, page: int) -> str:
        region_slug = _slugify_region(region)
        base = f"https://www.avito.ru/{region_slug}/vakansii"
        category_slug = _slugify_category(category)
        if category_slug:
            base = f"{base}/{category_slug}"
        params = {"p": page}
        if query:
            params["q"] = query
        return f"{base}?{urlencode(params, doseq=True)}"

    def _parse_listing(self, html_text: str, page_url: str) -> List[ListingCard]:
        soup = BeautifulSoup(html_text, "lxml")
        cards: List[ListingCard] = []
        nodes = soup.select('[data-marker="item"]')
        for node in nodes:
            link = node.select_one('a[data-marker="item-title"], a[data-marker="item/title"]')
            if not link or not link.get("href"):
                continue
            href = link.get("href")
            url = urljoin("https://www.avito.ru", href)
            ext_id = _extract_external_id(url)
            if not ext_id:
                continue
            title = link.get_text(strip=True)
            salary_node = node.select_one('[data-marker="item-price"], [data-marker="item/price"]')
            salary_text = salary_node.get_text(" ", strip=True) if salary_node else ""
            location_node = node.select_one('[data-marker="item-location"]')
            location_text = location_node.get_text(" ", strip=True) if location_node else ""
            address_node = node.select_one('[data-marker="item-address"]')
            address_raw = address_node.get_text(" ", strip=True) if address_node else None
            date_node = node.select_one('[data-marker="item-date"]')
            posted_at_raw = date_node.get_text(" ", strip=True) if date_node else None
            badge_text = " ".join(
                b.get_text(" ", strip=True) for b in node.select('[data-marker="item-badge"], [data-marker="item-badge/top"]')
            )
            is_promoted = bool(re.search(r"реклам|премиум|top", badge_text, re.IGNORECASE))
            is_featured = bool(re.search(r"топ|выдел", badge_text, re.IGNORECASE))
            snippet_node = node.select_one('[data-marker="item-snippet"], [data-marker="item-description"]')
            snippet_text = snippet_node.get_text(" ", strip=True) if snippet_node else None
            card = ListingCard(
                external_id=ext_id,
                url_listing=page_url,
                url_detail=url,
                title=title,
                salary_text=salary_text,
                location_text=location_text,
                address_raw=address_raw,
                posted_at_raw=posted_at_raw,
                is_promoted=is_promoted,
                is_featured=is_featured,
                snippet_text=snippet_text,
                raw_html=str(node),
                diagnostics={
                    "title_selector": ['a[data-marker="item-title"]'],
                    "salary_selector": ['[data-marker="item-price"]'],
                },
            )
            cards.append(card)
        self.stats["listing_cards"] += len(cards)
        return cards

    def _parse_detail(self, card: ListingCard, html_text: str, response_url: str) -> Dict[str, object]:
        soup = BeautifulSoup(html_text, "lxml")
        lang = soup.find("html")
        lang_code = lang.get("lang") if lang and lang.has_attr("lang") else "ru"

        title_node = soup.select_one('h1[data-marker="item-title"], h1[class*="title"], h1')
        title = title_node.get_text(strip=True) if title_node else card.title

        seller_name_node = soup.select_one('[data-marker="seller-info/name"], [data-marker="seller-link/linkText"]')
        employer_name = seller_name_node.get_text(strip=True) if seller_name_node else None
        employer_url_node = soup.select_one('[data-marker="seller-info/link"] a, a[data-marker="seller-link/link"]')
        employer_url = employer_url_node.get("href") if employer_url_node else None
        if employer_url and employer_url.startswith("//"):
            employer_url = "https:" + employer_url

        desc_node = soup.select_one('[data-marker="item-description/text"], [data-marker="item-description"]')
        description = desc_node.get_text("\n", strip=True) if desc_node else ""

        breadcrumbs = [
            _clean_text(item.get_text(" ", strip=True))
            for item in soup.select('[itemprop="itemListElement"] [itemprop="name"]')
        ]

        stats_text = " ".join(
            node.get_text(" ", strip=True)
            for node in soup.select('[data-marker="item-view/total"], [data-marker="item-view/favorites"]')
        )
        views_match = re.search(r"просмотр(?:ов|а)?[:\s]*(\d+)", stats_text, re.IGNORECASE)
        fav_match = re.search(r"избранн(?:ое|ом)?:?\s*(\d+)", stats_text, re.IGNORECASE)
        views_count = _safe_int(views_match.group(1) if views_match else None)
        favorites_count = _safe_int(fav_match.group(1) if fav_match else None)

        photo_nodes = soup.select('[data-marker="gallery/image"], [data-marker="slider-image"] img')
        photos_count = len(photo_nodes)

        params: Dict[str, str] = {}
        for li in soup.select('[data-marker="item-params/list-item"]'):
            label = li.select_one('[data-marker="item-params/list-item-label"], span[class*="-label"]')
            value_node = li.select_one('[data-marker="item-params/list-item-value"], span[class*="-value"]')
            if not value_node and label is None:
                text_val = _clean_text(li.get_text(" ", strip=True))
                if ":" in text_val:
                    key, val = text_val.split(":", 1)
                    params[key.strip().lower()] = val.strip()
                continue
            label_text = label.get_text(strip=True) if label else ""
            label_text = label_text.rstrip(":：")
            value_text = value_node.get_text(" ", strip=True) if value_node else ""
            params[label_text.strip().lower()] = value_text.strip()

        location_city = None
        location_area = None
        metro = None
        address_raw = card.address_raw or params.get("адрес", "") or params.get("адрес:", "")
        if params:
            for key, val in params.items():
                if "город" in key:
                    location_city = val
                if "район" in key:
                    location_area = val
                if "метро" in key:
                    metro = val

        if not location_city and card.location_text:
            location_city = card.location_text

        salary_candidates = []
        if params:
            for key, val in params.items():
                if re.search(r"зарплат|оплат", key):
                    salary_candidates.append(val)
        if card.salary_text:
            salary_candidates.append(card.salary_text)
        if description:
            first_line = description.splitlines()[0]
            salary_candidates.append(first_line)

        salary_from = salary_to = salary_currency = None
        salary_period = None
        for candidate in salary_candidates:
            if not candidate:
                continue
            minimum, maximum, currency = _extract_salary_numbers(candidate)
            period = _detect_salary_period(candidate)
            if minimum is None and maximum is None:
                continue
            salary_from = minimum if minimum is not None else salary_from
            salary_to = maximum if maximum is not None else salary_to
            if currency:
                salary_currency = currency
            if not salary_period and period:
                salary_period = period
            if salary_from is not None and salary_to is not None:
                break

        posted_at_raw = None
        posted_at = None
        time_nodes = soup.select('time[itemprop="datePublished"], time[itemprop="datePosted"], time')
        for node in time_nodes:
            raw_value = node.get("datetime") or node.get_text(strip=True)
            if not raw_value:
                continue
            posted_at_raw = raw_value
            try:
                posted_at = datetime.fromisoformat(raw_value)
                if not posted_at.tzinfo:
                    posted_at = posted_at.replace(tzinfo=self.config.timezone)
                break
            except ValueError:
                parsed = _parse_relative_datetime(raw_value, self.now)
                if parsed:
                    posted_at = parsed
                    break

        updated_at = None
        update_node = soup.select_one('[data-marker="item-view/update-time"], time[itemprop="dateModified"]')
        if update_node:
            raw_update = update_node.get("datetime") or update_node.get_text(strip=True)
            try:
                updated_at = datetime.fromisoformat(raw_update)
            except ValueError:
                parsed = _parse_relative_datetime(raw_update, self.now)
                updated_at = parsed

        schedule_hint = params.get("график работы") or params.get("график", "")
        schedule_hint = _normalize_schedule_hint(schedule_hint or description)

        employment_type = params.get("тип занятости") or params.get("занятость")
        experience_text = params.get("опыт работы") or params.get("опыт")
        experience_norm = None
        if experience_text:
            try:
                from fetch_vacancies import extract_experience

                experience_norm = extract_experience(experience_text)
            except Exception:  # pragma: no cover - defensive fallback
                experience_norm = None

        duties_raw = _extract_section_text(soup, ["обязанности", "что делать", "задачи"])
        requirements_raw = _extract_section_text(soup, ["требования", "кандидат"])
        conditions_raw = _extract_section_text(soup, ["условия", "мы предлагаем", "что предлагаем"])
        benefits = _gather_benefits(conditions_raw or "")

        contact_node = soup.select_one('[data-marker="seller-info/contact"]')
        contact = None
        if contact_node:
            contact = {
                "raw": contact_node.get_text(" ", strip=True),
            }

        contexts = []
        try:
            from fetch_vacancies import WorkingHoursContext, extract_working_hours

            if description:
                contexts.append(
                    WorkingHoursContext(
                        text=description,
                        url=response_url,
                        selector='[data-marker="item-description/text"]',
                        weight=10,
                    )
                )
            for key, val in params.items():
                if re.search(r"граф|смен|час", key) or re.search(r"граф|смен|час", val):
                    contexts.append(
                        WorkingHoursContext(
                            text=f"{key}: {val}",
                            url=response_url,
                            selector=f"params:{key}",
                            weight=6,
                        )
                    )
            working_hours = extract_working_hours(contexts)
        except Exception:  # pragma: no cover - defensive fallback
            working_hours = {
                "raw_matches": [description] if description else [],
                "normalized": None,
                "confidence": 0.0,
                "source": [],
                "schedule_hint": None,
            }

        if not schedule_hint:
            schedule_hint = working_hours.get("schedule_hint")

        html_scope = desc_node.decode() if desc_node else html_text
        snapshot_hash = _compute_hash(html_scope)

        diagnostics = {
            "detail_selectors": {
                "title": 'h1[data-marker="item-title"]',
                "params": '[data-marker="item-params/list-item"]',
                "description": '[data-marker="item-description/text"]',
            },
        }

        record = {
            "source": "avito",
            "external_id": card.external_id,
            "url_listing": card.url_listing,
            "url_detail": response_url or card.url_detail,
            "title": title,
            "employer_name": employer_name,
            "employer_profile_url": employer_url,
            "location_city": location_city,
            "location_area": location_area,
            "metro": metro,
            "address_raw": address_raw,
            "salary_from": salary_from,
            "salary_to": salary_to,
            "salary_currency": salary_currency,
            "salary_period": salary_period,
            "working_hours": working_hours,
            "schedule_hint": schedule_hint,
            "employment_type": employment_type,
            "experience_required": {
                "raw": experience_text,
                "normalized": experience_norm,
            },
            "duties_raw": duties_raw,
            "requirements_raw": requirements_raw,
            "conditions_raw": conditions_raw,
            "benefits": benefits,
            "contact": contact,
            "posted_at": posted_at,
            "posted_at_raw": posted_at_raw or card.posted_at_raw,
            "updated_at": updated_at,
            "views_count": views_count,
            "favorites_count": favorites_count,
            "is_promoted": card.is_promoted,
            "is_featured": card.is_featured,
            "category_path": breadcrumbs,
            "photos_count": photos_count,
            "html_snapshot_hash": snapshot_hash,
            "diagnostics": diagnostics,
            "lang": lang_code,
        }
        return record

    # ------------------------------------------------------------------
    def _persist_sample(self, card: ListingCard, detail_html: Optional[str], record: Dict[str, object]) -> None:
        for sample in self.qa_samples:
            if sample.get("external_id") == card.external_id:
                sample.update(
                    {
                        "listing_html": card.raw_html,
                        "detail_html": detail_html,
                        "record": record,
                    }
                )
                return
        if len(self.qa_samples) >= 10:
            return
        self.qa_samples.append(
            {
                "external_id": card.external_id,
                "listing_html": card.raw_html,
                "detail_html": detail_html,
                "record": record,
            }
        )

    def _maybe_dump_samples(self) -> None:
        if not self.config.qa_output_dir:
            return
        directory = self.config.qa_output_dir
        directory.mkdir(parents=True, exist_ok=True)
        for sample in self.qa_samples:
            ext_id = sample["external_id"]
            listing_path = directory / f"{ext_id}_listing.html"
            detail_path = directory / f"{ext_id}_detail.html"
            json_path = directory / f"{ext_id}_parsed.json"
            if "listing_html" in sample and sample["listing_html"]:
                listing_path.write_text(sample["listing_html"], encoding="utf-8")
            if sample.get("detail_html"):
                detail_path.write_text(str(sample["detail_html"]), encoding="utf-8")
            json_path.write_text(json.dumps(sample.get("record"), default=str, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    def collect(self) -> List[Dict[str, object]]:
        combos = []
        categories = self.config.categories or [None]
        for region in self.config.regions:
            for category in categories:
                for query in self.config.queries:
                    combos.append((region, category, query))

        min_allowed_date = self.config.min_allowed_date

        for region, category, query in combos:
            for page in range(1, self.config.max_pages_per_feed + 1):
                listing_url = self._build_listing_url(region, category, query, page)
                response = self._fetcher(listing_url)
                if not response or response.status_code != 200:
                    LOGGER.warning("Avito listing fetch failed: %s status=%s", listing_url, getattr(response, "status_code", None))
                    self.stats["detail_fail"] += 1
                    break
                cards = self._parse_listing(response.text, listing_url)
                if not cards:
                    break
                for card in cards:
                    posted_at = _parse_relative_datetime(card.posted_at_raw or "", self.now)
                    if posted_at and min_allowed_date and posted_at < min_allowed_date:
                        continue
                    detail_response = self._fetcher(card.url_detail)
                    detail_html = detail_response.text if detail_response and detail_response.status_code == 200 else None
                    diagnostics = {
                        "listing_url": card.url_listing,
                        "detail_url": card.url_detail,
                        "detail_status": getattr(detail_response, "status_code", None),
                    }
                    if detail_response is None or detail_response.status_code >= 400:
                        record = {
                            "source": "avito",
                            "external_id": card.external_id,
                            "url_listing": card.url_listing,
                            "url_detail": card.url_detail,
                            "title": card.title,
                            "is_active": False,
                            "last_seen_at": self.now,
                            "diagnostics": diagnostics,
                        }
                        self.records[card.external_id] = record
                        self.stats["detail_fail"] += 1
                        self._persist_sample(card, detail_html, record)
                        continue

                    detail_record = self._parse_detail(card, detail_response.text, detail_response.url or card.url_detail)
                    detail_record.update(
                        {
                            "is_active": True,
                            "last_seen_at": self.now,
                            "diagnostics": {**detail_record.get("diagnostics", {}), **diagnostics},
                        }
                    )

                    if self.config.only_with_salary and not (
                        detail_record.get("salary_from") or detail_record.get("salary_to")
                    ):
                        continue

                    existing = self.records.get(card.external_id)
                    if existing:
                        prev_promoted = bool(existing.get("is_promoted"))
                        prev_featured = bool(existing.get("is_featured"))
                        existing.update(detail_record)
                        existing["is_promoted"] = bool(detail_record.get("is_promoted") or prev_promoted)
                        existing["is_featured"] = bool(detail_record.get("is_featured") or prev_featured)
                        existing.setdefault("diagnostics", {}).setdefault("deduplicated", 0)
                        existing["diagnostics"]["deduplicated"] += 1
                    else:
                        self.records[card.external_id] = detail_record
                    self.stats["detail_success"] += 1
                    self._persist_sample(card, detail_response.text, detail_record)

        self._maybe_dump_samples()

        records = list(self.records.values())
        salary_count = sum(1 for rec in records if rec.get("salary_from") or rec.get("salary_to"))
        wh_count = sum(
            1
            for rec in records
            if rec.get("working_hours") and rec["working_hours"].get("raw_matches")
        )
        total = len(records) or 1
        LOGGER.info(
            "Avito stats: regions=%s queries=%s categories=%s records=%s salary_coverage=%.1f%% working_hours_coverage=%.1f%% errors=%s blocks=%s",
            self.config.regions,
            self.config.queries,
            self.config.categories,
            len(records),
            100.0 * salary_count / total,
            100.0 * wh_count / total,
            self.stats["detail_fail"],
            self.stats["blocked"],
        )
        return records


def collect_avito_vacancies(
    config: AvitoConfig,
    fetcher: Optional[Callable[[str], requests.Response]] = None,
    now: Optional[datetime] = None,
) -> List[Dict[str, object]]:
    """Convenience helper to collect Avito vacancies."""

    collector = AvitoCollector(config=config, fetcher=fetcher, now=now)
    return collector.collect()

