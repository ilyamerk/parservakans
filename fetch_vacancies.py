# parsers/fetch_vacancies.py  — версия с Avito (requests -> fallback на Playwright)
import argparse, requests, html
from typing import Tuple, Optional
from datetime import datetime, timedelta, timezone
import pandas as pd
import re, json, time
from pathlib import Path
from typing import List, Dict, Any
import requests, json, re
import math
from concurrent.futures import ThreadPoolExecutor, as_completed  # можно и локально внутри функции
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
# --- Avito/GR-only: polite HTTP helpers ---
import threading, random
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_AVITO_RPS = 0.35  # ~1 запрос каждые ~3 сек
_GR_RPS    = 0.50  # ~1 запрос каждые 2 сек

class _HostLimiter:
    def __init__(self):
        self._last = {}
        self._lock = threading.Lock()
    def wait(self, host: str, rps: float):
        gap = 1.0 / max(rps, 0.05)
        with self._lock:
            now = time.monotonic()
            last = self._last.get(host, 0.0)
            sleep = last + gap - now
            if sleep > 0:
                time.sleep(sleep + random.uniform(0.05, 0.25))
            self._last[host] = time.monotonic()

_LIM = _HostLimiter()

def _mk_polite_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    retry = Retry(
        total=3, connect=3, read=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s.mount("https://", adapter); s.mount("http://", adapter)
    return s

def _polite_get(sess, url: str, timeout: float, site: str):
    host = urlparse(url).netloc.lower()
    rps  = _GR_RPS if site == "gr" else _AVITO_RPS
    _LIM.wait(host, rps)
    try:
        r = sess.get(url, timeout=timeout)
        if r is not None and r.status_code == 429:
            ra = r.headers.get("Retry-After")
            delay = min(60.0, float(ra)) if (ra and ra.isdigit()) else 3.0
            time.sleep(delay + random.uniform(0, 0.3))
            _LIM.wait(host, rps)
            r = sess.get(url, timeout=timeout)
        return r
    except requests.RequestException as e:
        print(f"[NET] {e.__class__.__name__}: {url}")
        return None


_SESS = None
def _get_sess():
    global _SESS
    if _SESS is None:
        s = requests.Session()
        s.headers.update(HEADERS)
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _SESS = s
    return _SESS



from urllib.parse import urljoin, urlsplit, urlunsplit
AVITO_DEBUG = True  # временно включаем, чтобы видеть скрины/HTML в Exports/_debug


ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "Exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Playwright
try:
    from playwright.sync_api import sync_playwright
    _HAS_PW = True
except Exception:
    sync_playwright = None
    _HAS_PW = False

# Stealth: совместимость / no-op
try:
    import playwright_stealth as _ps
    if hasattr(_ps, "stealth_sync"):
        from playwright_stealth import stealth_sync as _stealth
    elif hasattr(_ps, "stealth"):
        from playwright_stealth import stealth as _stealth
    else:
        def _stealth(*_a, **_k): pass
except Exception:
    def _stealth(*_a, **_k): pass


_GR_CITY_SLUGS = {
    "москва": "moskva",
    "санкт-петербург": "sanktpeterburg",
    "санкт петербург": "sanktpeterburg",
    "спб": "sanktpeterburg",
    "екатеринбург": "ekaterinburg",
    "новосибирск": "novosibirsk",
    "казань": "kazan",
    "нижний новгород": "nizhnijnovgorod",
    "самара": "samara",
    "омск": "omsk",
    "ростов-на-дону": "rostovnadonu",
    "уфа": "ufa",
    "красноярск": "krasnojarsk",
    "пермь": "perm",
    "воронеж": "voronezh",
    "волгоград": "volgograd",
}
def _gr_city_slug(city: str) -> str:
    return _GR_CITY_SLUGS.get((city or "").strip().lower(), "")


# нормализация/проверка URL карточки
_ITEM_RE = re.compile(r"^https?://(?:www\.)?avito\.ru/.*/vakansii/[^/?#]+-\d{6,}(?:[/?#].*)?$")
def _is_item(u: str) -> bool:
    return bool(_ITEM_RE.match((u or "").strip().lower()))

def _normalize_url(u: str) -> str:
    u = (u or "").strip()
    u = re.sub(r"^https?://m\.avito\.ru", "https://www.avito.ru", u)
    u = re.sub(r"^https?://avito\.ru", "https://www.avito.ru", u)
    u = re.sub(r"\?.*$", "", u)
    return u

def _to_mobile(u: str) -> str:
    p = urlsplit(u); host = (p.netloc or "").lower().replace("www.","")
    if not host.startswith("m."): host = "m."+host
    return urlunsplit(("https", host, p.path, "", ""))

def _collect_urls_from_page(page) -> set[str]:
    urls = set()
    # 1) DOM-якоря карточек
    for sel in ['a[data-marker="item-title"]', 'a[href*="/vakansii/"]']:
        try:
            for a in page.locator(sel).all():
                href = (a.get_attribute("href") or "").strip()
                if not href:
                    continue
                if href.startswith("/"):
                    href = urljoin("https://www.avito.ru", href)
                if "/vakansii/" in href and re.search(r"-\d{6,}(?:[/?#].*)?$", href):
                    urls.add(_normalize_url(href))
        except Exception:
            pass

    # 2) JSON в <script> (initial state)
    try:
        for sc in page.locator("script").all()[:40]:
            try:
                txt = sc.text_content() or ""
            except Exception:
                continue
            for path in re.findall(r'"urlPath":"(\/[^"]+\/vakansii\/[^"]+?-\d{6,})"', txt):
                urls.add(_normalize_url(urljoin("https://www.avito.ru", path)))
            for rel in re.findall(r'"url":"(\/\/www\.avito\.ru\/[^"]+)"', txt):
                urls.add(_normalize_url("https:" + rel.encode("utf-8").decode("unicode_escape")))
    except Exception:
        pass
    return urls

def _avito_listing_desktop(query: str, city_slug: str, pages: int, pause: float,
                           headful: bool = False, state_path: str = "avito_state.json") -> list[str]:
    if not _HAS_PW:
        print("[Avito] Playwright недоступен — установи: pip install playwright && python -m playwright install chromium")
        return []

    q = query.strip()
    tag = _slugify(q) or "vakansii"


    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not headful)
        ctx_kwargs = dict(
            locale="ru-RU",
            timezone_id="Europe/Moscow",
            user_agent=HEADERS["User-Agent"],
            viewport={"width": 1366, "height": 860},
            device_scale_factor=1.25,
        )
        # подхват куки, если уже есть
        try:
            if state_path and Path(state_path).exists():
                ctx_kwargs["storage_state"] = state_path
        except Exception:
            pass

        ctx = browser.new_context(**ctx_kwargs)
        page = ctx.new_page()
        titles: dict[str, str] = {}
        try: _stealth(page)
        except Exception: pass

        # --- WARMUP: заходим на главную и делаем «человеческие» действия ---
        try:
            page.goto("https://www.avito.ru/", wait_until="domcontentloaded", timeout=45000)
            title_low = (page.title() or "").lower()
            if "доступ ограничен" in title_low or "access denied" in title_low:
                print("[Avito] доступ ограничен этим IP — пропускаю Avito.")
                return []

            # куки/регион/подсказки на главной
            for sel in [
                'button[data-marker="location/region-ok"]',
                'button[data-marker="popup-location/save-button"]',
                'button[aria-label="Принять"]',
                'button:has-text("Принять")',
                'button:has-text("Да")',
                'button:has-text("Понятно")',
                'button:has-text("Продолжить")',
            ]:
                try:
                    if page.locator(sel).count() > 0:
                        page.locator(sel).first.click(timeout=1200)
                        page.wait_for_timeout(350)
                except Exception:
                    pass

            # немного «пожить» на странице
            page.wait_for_timeout(800)
            page.mouse.wheel(0, 800)
            page.wait_for_timeout(400)
        except Exception:
            pass

        # Несколько десктопных URL — разные антиботы и разная верстка
        bases = [
            f"https://www.avito.ru/{city_slug}/vakansii?q={q}&s=104",                # поиск
            f"https://www.avito.ru/{city_slug}/vakansii/tag/{tag}?localPriority=0",  # тег
            f"https://www.avito.ru/{city_slug}/vakansii?cd=1&q={q}",                 # альтернативный поиск
            f"https://www.avito.ru/all/vakansii?q={q}&s=104",                        # fallback на /all
        ]

        all_urls = set()
        print("[Avito][playwright] listing start (desktop)")
        for p in range(1, pages + 1):
            for base in bases:
                url = base + (("&p=%d" % p) if "?" in base else ("?p=%d" % p))
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=45000)

                    # кликаем куки/баннеры и во всех iframe (если всплыли)
                    def _try_click_all():
                        selectors = [
                            'button[aria-label="Принять"]', 'button:has-text("Принять")',
                            'button:has-text("Понятно")', 'button:has-text("Продолжить")',
                            'button:has-text("Да")', 'button:has-text("Согласен")',
                            '[role="button"]:has-text("OK")',
                        ]
                        # main
                        for sel in selectors:
                            try:
                                if page.locator(sel).count() > 0:
                                    page.locator(sel).first.click(timeout=1200)
                                    page.wait_for_timeout(250)
                            except Exception:
                                pass
                        # iframes
                        try:
                            for fr in page.frames:
                                for sel in selectors:
                                    try:
                                        if fr.query_selector(sel):
                                            fr.click(sel, timeout=1200)
                                            page.wait_for_timeout(250)
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                    _try_click_all()

                    # ждём, пока на странице появятся **карточки** (несколько вариантов верстки)
                    candidate_lists = [
                        'a[data-qa="vacancy-title"]',
                        'a.vacancy-item__title',
                        'article a[href*="vacan"]',  # очень широкий
                        'main a[href*="vacan"]',
                    ]
                    appeared = False
                    for sel in candidate_lists:
                        try:
                            page.wait_for_selector(sel, timeout=6000)
                            if page.locator(sel).count() > 0:
                                appeared = True
                                break
                        except Exception:
                            continue

                    # плавно прокрутим (ленивая подгрузка)
                    last_h = 0
                    for _ in range(24):
                        page.mouse.wheel(0, 1400)
                        page.wait_for_timeout(220)
                        try:
                            h = page.evaluate("document.body.scrollHeight")
                        except Exception:
                            h = 0
                        if h <= last_h:
                            break
                        last_h = h

                    # куки/регион/подсказки
                    for sel in [
                        'button[data-marker="location/region-ok"]',
                        'button[data-marker="popup-location/save-button"]',
                        'button[aria-label="Принять"]',
                        'button:has-text("Принять")',
                        'button:has-text("Да")',
                        'button:has-text("Продолжить")',
                        'button:has-text("Понятно")',
                    ]:
                        try:
                            if page.locator(sel).count() > 0:
                                page.locator(sel).first.click(timeout=1200)
                                page.wait_for_timeout(350)
                        except Exception:
                            pass

                    # Ждём именно карточки
                    try:
                        page.wait_for_selector('a[data-marker="item-title"], a[href*="/vakansii/"]',
                                               timeout=12000)
                    except Exception:
                        page.wait_for_load_state("networkidle", timeout=8000)

                    # Прокрутка до упора (пока растёт высота)
                    last_h = 0
                    for _ in range(24):
                        page.mouse.wheel(0, 1600)
                        page.wait_for_timeout(250)
                        try:
                            h = page.evaluate("document.body.scrollHeight")
                        except Exception:
                            h = 0
                        if h <= last_h:
                            break
                        last_h = h

                    before = len(all_urls)
                    all_urls |= _collect_urls_from_page(page)

                    # Диагностика: скрин + HTML
                    _save_debug(page, f"avito_w_p{p}")

                    html_low = (page.content() or "").lower()
                    if ("captcha" in html_low) or ("подозрительная активность" in html_low) or ("доступ ограничен" in html_low):
                        print("[Avito][hint] Похоже, капча/блокировка. Запусти с --avito_headful и реши капчу в окне.")

                    print(f"[Avito][PW] page={page.url} title={page.title()!r} +{len(all_urls)-before} urls")
                except Exception as e:
                    print(f"[Avito][PW] page fail: {e}")
                page.wait_for_timeout(int(max(0.3, pause) * 1000))

        # сохраняем state (куки) на будущие запуски
        try:
            if state_path:
                ctx.storage_state(path=state_path)
        except Exception:
            pass

        browser.close()

    urls = [u for u in map(_normalize_url, all_urls) if _is_item(u)]
    print(f"[Avito] listing URLs (cards only): {len(urls)}")
    return sorted(set(urls))


def _normalize_avito_url(u: str) -> str:
    u = (u or "").strip()
    u = re.sub(r"^https?://m\.avito\.ru", "https://www.avito.ru", u)
    u = re.sub(r"^https?://avito\.ru", "https://www.avito.ru", u)
    u = re.sub(r"\?.*$", "", u)
    return u

_TR = str.maketrans({"а":"a","б":"b","в":"v","г":"g","д":"d","е":"e","ё":"e","ж":"zh","з":"z","и":"i","й":"y","к":"k","л":"l","м":"m","н":"n","о":"o","п":"p","р":"r","с":"s","т":"t","у":"u","ф":"f","х":"h","ц":"c","ч":"ch","ш":"sh","щ":"sch","ъ":"","ы":"y","ь":"","э":"e","ю":"yu","я":"ya"})
def _slugify_tag(q: str) -> str:
    s = (q or "").strip().lower().translate(_TR)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "vakansii"


def _extract_item_urls_from_html(html: str, base: str) -> List[str]:
    """Достаёт URL карточек из html и вложенных JSON кусков."""
    urls = set()

    # 1) Мгновенно: прямые <a href="/.../vakansii/...-123456789">
    for href in re.findall(r'href="(\/[^"]+\/vakansii\/[^"]+?-\d{6,})"', html):
        urls.add(urljoin("https://www.avito.ru", href))

    # 2) Любые абсолютные ссылки в тексте
    for absu in re.findall(r'(https://www\.avito\.ru/[^"\']+/vakansii/[^"\']+?-\d{6,})', html):
        urls.add(_normalize_avito_url(absu))

    # 3) Вырезаем JSON из <script data-state> / window.__initial_state__ и пробегаем regex заново
    try:
        soup = BeautifulSoup(html, "lxml") if _HAS_BS4 else None
        if soup:
            for sc in soup.find_all("script"):
                txt = sc.string or sc.text or ""
                if not txt:
                    continue
                for path in re.findall(r'"urlPath":"(\/[^"]+\/vakansii\/[^"]+?-\d{6,})"', txt):
                    urls.add(urljoin("https://www.avito.ru", path))
                for rel in re.findall(r'"url":"(\/\/www\.avito\.ru\/[^"]+)"', txt):
                    urls.add("https:" + rel.encode("utf-8").decode("unicode_escape"))
    except Exception:
        pass

    # нормализуем и отфильтруем
    good = []
    for u in urls:
        u = _normalize_avito_url(u)
        if _is_avito_item_url(u) and "/vakansii/tag/" not in u:
            good.append(u)
    return list(set(good))



_WORKERS = 8
_TIMEOUT = 8.0


# --- helper: deduplicate sequence by key ---
def _dedup(seq, key=lambda x: x):
    seen = set()
    out = []
    for item in seq:
        try:
            k = key(item)
        except Exception:
            k = item
        if k in seen:
            continue
        seen.add(k)
        out.append(item)
    return out


import json

# ---- optional bs4 for HTML (hh + gorodrabot + avito via requests) ----
try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    BeautifulSoup = None
    _HAS_BS4 = False

# --- заголовки поумнее, чтобы меньше ловить 403/429 ---
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.7,en;q=0.6",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "DNT": "1",
}
# --- ru->latin slug для Avito tag ---
_TR = str.maketrans({
    "а":"a","б":"b","в":"v","г":"g","д":"d","е":"e","ё":"e","ж":"zh","з":"z","и":"i","й":"y","к":"k",
    "л":"l","м":"m","н":"n","о":"o","п":"p","р":"r","с":"s","т":"t","у":"u","ф":"f","х":"h","ц":"c",
    "ч":"ch","ш":"sh","щ":"sch","ъ":"","ы":"y","ь":"","э":"e","ю":"yu","я":"ya"
})
def _slugify(s: str) -> str:
    s = (s or "").strip().lower().translate(_TR)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


TEMPLATE_COLS = [
    "Должность","Работодатель","Дата публикации",
    "ЗП от (т.р.)","ЗП до (т.р.)",
    "Средний совокупный доход при графике 2/2 по 12 часов","В час","Ставка (расчётная) в час, ₽","Длительность \nсмены",
    "Требуемый\nопыт","Труд-во","График","Частота \nвыплат","Льготы","Обязаности","Ссылка","Примечание"
]

GROSS_NOTE_TEXT = "До вычета налога"

_GROSS_POS_PATTERNS = [
    re.compile(r"до\s*вычета(?:\s*налога)?", re.IGNORECASE),
    re.compile(r"до\s*вычета\s*ндфл", re.IGNORECASE),
    re.compile(r"до\s*удержания(?:\s*ндфл)?", re.IGNORECASE),
    re.compile(r"до\s*уплаты\s*налогов", re.IGNORECASE),
    re.compile(r"\bgross\b", re.IGNORECASE),
    re.compile(r"\bбрутто\b", re.IGNORECASE),
]

_GROSS_NEG_PATTERNS = [
    re.compile(r"на\s*руки", re.IGNORECASE),
    re.compile(r"после\s*вычета", re.IGNORECASE),
    re.compile(r"\bnet\b", re.IGNORECASE),
    re.compile(r"\bчистыми\b", re.IGNORECASE),
]

_NBSP_RE = re.compile(r"\u00a0", re.UNICODE)


def _normalize_rate_text(text: str) -> str:
    if not text:
        return ""
    cleaned = _NBSP_RE.sub(" ", str(text))
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


_RATE_LOWER_RE = re.compile(r"\s+", re.UNICODE)

_HOURLY_MARKERS = [
    r"\bв\s*час\b",
    r"/\s*ч(?:ас)?\b",
    r"\bчасова[яе]\s*ставка\b",
    r"₽\s*/\s*ч",
    r"руб\.?\s*/\s*ч",
    r"\brur/hour\b",
    r"\brub/hour\b",
    r"\brub/hr\b",
    r"\buah/hour\b",
    r"\beur/hour\b",
    r"\busd/hour\b",
    r"\$\s*/\s*hour",
    r"\$\s*/\s*hr",
    r"\bper[-\s]?hour\b",
    r"\bhourly\b",
]

_SHIFT_MARKERS = [
    "за смену",
    "/смена",
    "/смен",
    "₽/смена",
    "руб/смена",
    "руб./смена",
    "сменная ставка",
    "per shift",
    "per-shift",
    "shift pay",
]

_NEGATIVE_SHIFT_HINTS = ["оклад", "месяц", "в день"]

_SHIFT_PATTERNS_RAW = [
    r"(?:€|\$|₽|руб\.?|rur)\s*(\d[\d\s]{0,12}(?:[.,]\d{1,2})?)\s*(?:–|-|до\s*)?(\d[\d\s]{0,12}(?:[.,]\d{1,2})?)?\s*(?:/|\s*)(?:смен[аи]|per\s*shift|per-shift|shift\s*pay)",
    r"(?:от\s*)?(\d[\d\s]{0,12}(?:[.,]\d{1,2})?)\s*(?:–|-|до\s*)?(\d[\d\s]{0,12}(?:[.,]\d{1,2})?)?\s*(?:₽|руб\.?|rur|€|\$)?\s*(?:/|\s*)(?:смен[аи]|per\s*shift|per-shift|shift\s*pay)",
]

_SHIFT_GENERIC_RE = re.compile(r"смен[^\n\r]{0,40}(?:\d|₽|руб|rur|\$|€)", re.IGNORECASE)

_HOURLY_MARKER_RES = [re.compile(m, re.IGNORECASE) for m in _HOURLY_MARKERS]
_SHIFT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _SHIFT_PATTERNS_RAW]


def _norm_for_match(text: str) -> str:
    norm = _normalize_rate_text(text).lower()
    return _RATE_LOWER_RE.sub(" ", norm)


def is_hourly_rate(text: str) -> bool:
    norm = _norm_for_match(text)
    if not norm:
        return False
    return any(p.search(norm) for p in _HOURLY_MARKER_RES)


def is_shift_rate(text: str) -> bool:
    norm = _norm_for_match(text)
    if not norm:
        return False
    if any(hint in norm for hint in _NEGATIVE_SHIFT_HINTS) and "смен" not in norm:
        return False
    if any(marker in norm for marker in _SHIFT_MARKERS):
        return True
    if "смен" in norm and _SHIFT_GENERIC_RE.search(norm):
        return True
    return False


_CURRENCY_SYMBOLS = {
    "RUB": "₽",
    "USD": "$",
    "EUR": "€",
    "UAH": "₴",
    "KZT": "₸",
    "BYN": "Br",
    "GBP": "£",
}


def _currency_symbol_from_code(code: Optional[str], fallback: Optional[str] = None) -> Optional[str]:
    if not code:
        return fallback
    return _CURRENCY_SYMBOLS.get(code.upper(), fallback)


def extract_shift_rate(text: str) -> dict | None:
    segment = _normalize_rate_text(text)
    if not segment:
        return None
    candidates: list[str] = []
    for pattern in _SHIFT_PATTERNS:
        for match in pattern.finditer(segment):
            raw_value = match.group(0).strip()
            if raw_value and raw_value not in candidates:
                candidates.append(raw_value)
    if not candidates:
        fragment = _find_shift_fragment(segment)
        if fragment:
            candidates.append(fragment)
    for raw_value in candidates:
        info = extract_rate(raw_value)
        if not info:
            continue
        currency_match = _CURRENCY_TOKEN_RE.search(raw_value)
        currency_token = currency_match.group(0) if currency_match else None
        currency_code = info.get("currency") or _currency_code(currency_token)
        if currency_code is None and "смен" in raw_value.lower():
            currency_code = "RUB"
        symbol = _currency_symbol_from_code(currency_code, fallback=currency_token.strip() if currency_token else None)
        value_display = str(info.get("value") or "").strip()
        if currency_code == "RUB" and symbol and "₽" not in value_display:
            value_display = f"{value_display} {symbol}".strip()
        elif currency_code in {"USD", "EUR", "GBP"} and symbol and not value_display.startswith(symbol):
            value_display = f"{symbol}{value_display}"
        elif symbol and currency_code not in {"USD", "EUR", "GBP"} and symbol not in value_display:
            value_display = f"{value_display} {symbol}".strip()
        info["value"] = value_display or raw_value
        info["currency"] = currency_code
        info["raw"] = raw_value
        return info
    if candidates:
        raw_value = candidates[0]
        return {
            "value": raw_value,
            "min": None,
            "max": None,
            "currency": None,
            "raw": raw_value,
        }
    return None


_RATE_NUMBER_RE = re.compile(r"\d[\d\s]*(?:[\.,]\d+)?")
_CURRENCY_TOKEN_RE = re.compile(
    r"₽|руб\.?|rur|rub|uah|грн|₴|€|eur|usd|\$|kzt|₸|byn|br|£|gbp",
    re.IGNORECASE,
)


def _parse_number_token(token: str) -> Optional[float]:
    cleaned = token.replace(" ", "").replace("\u202f", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _currency_code(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    t = token.strip().lower()
    if t in {"₽", "руб", "руб.", "rur", "rub"}:
        return "RUB"
    if t in {"$", "usd"}:
        return "USD"
    if t in {"€", "eur"}:
        return "EUR"
    if t in {"uah", "грн", "₴"}:
        return "UAH"
    if t in {"kzt", "₸"}:
        return "KZT"
    if t in {"byn", "br"}:
        return "BYN"
    if t in {"£", "gbp"}:
        return "GBP"
    return token.strip().upper()


def _replace_currency_words(value: str) -> str:
    repl = [
        (r"(?i)\bруб\.?\b", "₽"),
        (r"(?i)\brub\b", "₽"),
        (r"(?i)\brur\b", "₽"),
        (r"(?i)\bгрн\b", "₴"),
        (r"(?i)\buah\b", "₴"),
        (r"(?i)\busd\b", "$"),
        (r"(?i)\beur\b", "€"),
        (r"(?i)\bkzt\b", "₸"),
        (r"(?i)\bbyn\b", "Br"),
        (r"(?i)\bgbp\b", "£"),
        (r"(?i)\bbr\b", "Br"),
    ]
    out = value
    for pattern, repl_val in repl:
        out = re.sub(pattern, repl_val, out)
    return out


def _clean_units(value: str) -> str:
    value = re.sub(
        r"(?i)(?:\s*(?:/|в|per[-\s]?|за)\s*(?:час|ч|hour|hr|смен[ауы]|shift|сменн\w*\s*ставка))+",
        "",
        value,
    )
    return value.strip()


def _select_number_matches(text: str, matches: list[re.Match[str]]) -> list[re.Match[str]]:
    relevant: list[re.Match[str]] = []
    for m in matches:
        start, end = m.span()
        before = text[max(0, start - 6):start]
        after = text[end:end + 8]
        if _CURRENCY_TOKEN_RE.search(before) or re.match(r"\s*(?:₽|\$|€|£|₴|₸)", after):
            relevant.append(m)
            continue
        if re.search(r"(?i)(?:руб\.?|rur|rub|uah|грн|eur|usd|kzt|byn|gbp|br)$", before):
            relevant.append(m)
            continue
        if re.match(r"(?i)\s*(?:/|per[-\s]?|в|за)\s*(?:час|ч|hour|hr|смен|shift)", after):
            relevant.append(m)
            continue
    return relevant


def extract_rate(text: str) -> dict | None:
    segment = _normalize_rate_text(text)
    if not segment:
        return None
    matches = list(_RATE_NUMBER_RE.finditer(segment))
    if not matches:
        return None
    selected = _select_number_matches(segment, matches)
    has_range_words = "от" in segment.lower() and "до" in segment.lower()
    if not selected:
        if has_range_words and len(matches) >= 2:
            selected = matches[:2]
        elif len(matches) >= 2:
            selected = matches[-2:]
        else:
            selected = matches[-1:]
    elif len(selected) == 1 and has_range_words and len(matches) >= 2:
        selected = matches[:2]
    if not selected:
        return None
    numbers: list[float] = []
    for m in selected:
        val = _parse_number_token(m.group(0))
        if val is None:
            continue
        numbers.append(val)
    if not numbers:
        return None
    min_val = numbers[0]
    max_val = numbers[1] if len(numbers) > 1 else numbers[0]
    if min_val is not None and max_val is not None and min_val > max_val:
        min_val, max_val = max_val, min_val
    currency_match = _CURRENCY_TOKEN_RE.search(segment)
    currency = _currency_code(currency_match.group(0)) if currency_match else None
    start = min(m.start() for m in selected)
    end = max(m.end() for m in selected)
    prefix_slice = segment[max(0, start - 6):start]
    prefix_curr = re.search(r"(?i)(₽|\$|€|£|₴|₸|руб\.?|rur|rub|uah|грн|usd|eur|kzt|byn|gbp|br)\s*$", prefix_slice)
    if prefix_curr:
        start = max(0, start - (len(prefix_slice) - prefix_curr.start()))
    elif has_range_words:
        word_slice_start = max(0, start - 3)
        prefix_word = segment[word_slice_start:start]
        if prefix_word.lower() == "от ":
            start = word_slice_start
    while end < len(segment) and segment[end] not in ".,;!\n\r":
        end += 1
    raw_value = segment[start:end].strip()
    cleaned_value = _clean_units(raw_value)
    cleaned_value = _replace_currency_words(cleaned_value)
    cleaned_value = re.sub(r"(?<=\d)\s*[-–—]\s*(?=\d)", "–", cleaned_value)
    cleaned_value = re.sub(r"\s+", " ", cleaned_value).strip()
    if not cleaned_value:
        cleaned_value = raw_value
    return {
        "value": cleaned_value,
        "min": min_val,
        "max": max_val,
        "currency": currency,
        "raw": raw_value,
    }


def _collect_text_chunks(value) -> list[str]:
    texts: list[str] = []
    if isinstance(value, str):
        val = value.strip()
        if val:
            texts.append(val)
    elif isinstance(value, dict):
        for key, val in value.items():
            if key == "gross":
                if val is True:
                    texts.append("gross")
                elif val is False:
                    texts.append("net")
                continue
            texts.extend(_collect_text_chunks(val))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            texts.extend(_collect_text_chunks(item))
    return texts


def _gross_text_blob(*parts) -> str:
    out: list[str] = []
    for part in parts:
        out.extend(_collect_text_chunks(part))
    return " ".join(out)


def _rate_text_blob(*parts) -> str:
    out: list[str] = []
    for part in parts:
        out.extend(_collect_text_chunks(part))
    return " ".join(out)


def _find_shift_fragment(text: str) -> str:
    cleaned = _normalize_rate_text(text)
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    for marker in _SHIFT_MARKERS:
        idx = lowered.find(marker)
        if idx != -1:
            left = idx
            while left > 0 and cleaned[left - 1] not in ".,;!\n\r":
                left -= 1
            right = idx + len(marker)
            while right < len(cleaned) and cleaned[right] not in ".,;!\n\r":
                right += 1
            fragment = cleaned[left:right].strip()
            if fragment:
                return fragment
    idx = lowered.find("смен")
    while idx != -1:
        left = idx
        while left > 0 and cleaned[left - 1] not in ".,;!\n\r":
            left -= 1
        right = idx
        while right < len(cleaned) and cleaned[right] not in ".,;!\n\r":
            right += 1
        fragment = cleaned[left:right].strip()
        if fragment:
            return fragment
        idx = lowered.find("смен", idx + 4)
    return ""


def collect_shift_rate_rows(rows: list[dict]) -> list[dict]:
    collected: list[dict] = []
    seen_urls: set[str] = set()
    for row in rows or []:
        url = (row.get("Ссылка") or row.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        text_blob = _rate_text_blob(
            row.get("__rate_text"),
            row.get("__text"),
            row.get("Должность"),
            row.get("Обязаности"),
            row.get("Льготы"),
            row.get("График"),
            row.get("Труд-во"),
        )
        if not text_blob:
            continue
        if not is_shift_rate(text_blob):
            continue
        rate_info = extract_shift_rate(text_blob)
        if rate_info:
            value = str(rate_info.get("value") or "").strip()
            raw_value = str(rate_info.get("raw") or "").strip()
        else:
            value = ""
            raw_value = _find_shift_fragment(text_blob)
        collected.append({
            "value": value or raw_value,
            "min": rate_info.get("min") if rate_info else None,
            "max": rate_info.get("max") if rate_info else None,
            "currency": rate_info.get("currency") if rate_info else None,
            "raw": raw_value,
            "url": url,
        })
        seen_urls.add(url)
    return collected


HOURS_PER_SHIFT_DEFAULT = 12

SHIFT_MAP_FIXED = {
    "5/2": 23,
    "2/2": 15,
    "3/3": 15,
    "4/2": 20,
    "6/1": 26,
    "2/1": 20,
    "1/2": 10,
    "1/3": 8,
    "7/7": 15,
    "15/15": 15,
    "30/30": 15,
    "12/36": 15,
    "24/48": 10,
}

SYNONYMS = {
    "пятидневка": "5/2",
    "сутки/двое": "24/48",
    "сутки через двое": "24/48",
    "вахта 15/15": "15/15",
    "вахта 30/30": "30/30",
}


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().replace("\xa0", " ")).strip()


_HOUR_CONTEXT_KEYWORDS = (
    "смен",
    "рабоч",
    "график",
    "длитель",
    "продолж",
    "раб. в",
    "раб.в",
    "рабочее время",
    "время работ",
    "рабочий день",
    "раб день",
    "рабочая смен",
    "кажд",
    "в смене",
    "смена дл",
    "смены дл",
    "по смен",
    "по графику",
)

_HOUR_RANGE_RE = re.compile(r"(\d{1,2})\s*(?:[-–]|до)\s*(\d{1,2})\s*(?:час(?:а|ов)?|ч)\b")
_HOUR_VALUE_RE = re.compile(r"(\d{1,2})(?:[.,](\d{1,2}))?\s*(?:час(?:а|ов)?|ч)\b")
_HOUR_COMPOUND_RE = re.compile(r"(\d{1,2})\s*[-–]?\s*час(?:ов|овая|овые|овой|ами|а)\b")
_TIME_RANGE_RE = re.compile(
    r"с\s*(\d{1,2})(?::(\d{2}))?\s*до\s*(\d{1,2})(?::(\d{2}))?"
)


def _parse_hour_value(value: str) -> Optional[float]:
    try:
        return float(value.replace(",", "."))
    except Exception:
        return None


def _context_has_keywords(before: str, after: str) -> bool:
    context = f"{before[-40:]} {after[:40]}"
    for kw in _HOUR_CONTEXT_KEYWORDS:
        if kw in context:
            return True
    return False


def detect_hours_per_shift(text: str, explicit_field: Optional[str]) -> float:
    candidates: list[float] = []
    if explicit_field:
        try:
            match = re.search(r"\d{1,2}(?:[.,]\d{1,2})?", str(explicit_field))
            if match:
                parsed = _parse_hour_value(match.group())
                if parsed is not None and 4 <= parsed <= 24:
                    candidates.append(parsed)
        except Exception:
            pass

    t = normalize_text(text)
    if t:
        for m in _HOUR_RANGE_RE.finditer(t):
            for idx in (1, 2):
                parsed = _parse_hour_value(m.group(idx))
                if parsed is not None and 4 <= parsed <= 24:
                    candidates.append(parsed)
        for m in _HOUR_VALUE_RE.finditer(t):
            parsed = _parse_hour_value(m.group(1) + ("." + m.group(2) if m.group(2) else ""))
            if parsed is None or not (4 <= parsed <= 24):
                continue
            before = t[max(0, m.start() - 40) : m.start()]
            after = t[m.end() : m.end() + 40]
            if _context_has_keywords(before, after):
                candidates.append(parsed)
        for m in _HOUR_COMPOUND_RE.finditer(t):
            parsed = _parse_hour_value(m.group(1))
            if parsed is not None and 4 <= parsed <= 24:
                before = t[max(0, m.start() - 40) : m.start()]
                after = t[m.end() : m.end() + 40]
                if _context_has_keywords(before, after):
                    candidates.append(parsed)
        for m in _TIME_RANGE_RE.finditer(t):
            start_h = int(m.group(1))
            start_m = int(m.group(2) or 0)
            end_h = int(m.group(3))
            end_m = int(m.group(4) or 0)
            start = start_h + start_m / 60
            end = end_h + end_m / 60
            duration = end - start
            if duration <= 0:
                duration += 24
            if 4 <= duration <= 24:
                candidates.append(round(duration, 1))

    for candidate in candidates:
        if 4 <= candidate <= 24:
            return candidate
    return float(HOURS_PER_SHIFT_DEFAULT)


def parse_schedule_to_key(text: str) -> Optional[str]:
    t = normalize_text(text)
    if not t:
        return None
    for key, value in SYNONYMS.items():
        if key in t:
            return value
    if "день/ночь" in t or "день ночь" in t:
        return "2/2"
    m = re.search(r"\b(\d{1,2})\s*/\s*(\d{1,2})\b", t)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        if 1 <= x <= 30 and 0 <= y <= 72:
            return f"{x}/{y}"
    m = re.search(r"\b(\d{1,2})\s*(?:через|-|\/\s*через)\s*(\d{1,2})\b", t)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        if 1 <= x <= 30 and 0 <= y <= 72:
            return f"{x}/{y}"
    return None


def estimate_shifts_per_month(schedule_text: str, hours_mode: bool = False, x: int | None = None, y: int | None = None) -> int:
    if schedule_text in SHIFT_MAP_FIXED:
        return SHIFT_MAP_FIXED[schedule_text]
    if x is None or y is None:
        m = re.match(r"(\d{1,2})/(\d{1,2})$", schedule_text or "")
        if m:
            x, y = int(m.group(1)), int(m.group(2))
    if x is not None and y is not None and (x + y) > 0:
        if not hours_mode and x <= 24 and y >= 24:
            hours_mode = True
        if hours_mode:
            shifts = round((30.44 * 24) / (x + y))
        else:
            shifts = round((30.44 * x) / (x + y))
        return max(int(shifts), 1)
    return 15


def _value_is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, (int, bool)):
        return False
    text = str(value).strip().lower()
    return text == "" or text == "nan" or text == "none"


def compute_hourly_rate(vac: dict) -> Optional[float]:
    sal_from_th = vac.get("ЗП от (т.р.)")
    if sal_from_th in (None, ""):
        sal_from_th = vac.get("salary_from_thousands")
    if sal_from_th in (None, ""):
        return None
    try:
        sal_from_th = float(str(sal_from_th).replace(",", "."))
    except Exception:
        return None
    if sal_from_th <= 0:
        return None

    hours_field = (
        vac.get("Длительность смены")
        or vac.get("Смена, часов")
        or vac.get("Длительность \nсмены")
    )

    merged_text_parts = [
        vac.get("Заголовок"),
        vac.get("Должность"),
        vac.get("Описание"),
        vac.get("Обязаности"),
        vac.get("Условия"),
        vac.get("График"),
        vac.get("Режим"),
        vac.get("Сменность"),
        vac.get("Льготы"),
        vac.get("Зарплата"),
        vac.get("Компенсация"),
    ]
    merged_text = " ".join(str(p) for p in merged_text_parts if p)
    hours_per_shift = detect_hours_per_shift(merged_text, hours_field)

    schedule_field = vac.get("График") or vac.get("Режим") or vac.get("Сменность") or ""
    schedule_source = f"{schedule_field} {merged_text}".strip()
    schedule_key = parse_schedule_to_key(schedule_source)
    if schedule_key:
        m = re.match(r"(\d{1,2})/(\d{1,2})", schedule_key)
        x = int(m.group(1)) if m else None
        y = int(m.group(2)) if m else None
        hours_mode = False
        if x is not None and y is not None:
            hours_mode = x <= 24 and y >= 24
            if hours_mode and (hours_per_shift == HOURS_PER_SHIFT_DEFAULT or hours_per_shift is None):
                if 4 <= x <= 24:
                    hours_per_shift = float(x)
        shifts_per_month = estimate_shifts_per_month(schedule_key, hours_mode=hours_mode, x=x, y=y)
    else:
        shifts_per_month = 15

    if hours_per_shift is None or hours_per_shift <= 0 or shifts_per_month <= 0:
        return None

    hourly = (sal_from_th * 1000.0) / (float(hours_per_shift) * shifts_per_month)
    return round(hourly, 1)


def apply_computed_hourly_rates(rows: list[dict]) -> None:
    for vac in rows or []:
        if not isinstance(vac, dict):
            continue
        hourly = compute_hourly_rate(vac)
        if hourly is None:
            vac.setdefault("Ставка (расчётная) в час, ₽", None)
            continue
        current = vac.get("В час")
        if _value_is_blank(current):
            vac["В час"] = hourly
        vac["Ставка (расчётная) в час, ₽"] = hourly


def is_gross_salary(text: str) -> bool:
    """Return True if the text contains markers that salary is gross."""
    if not text:
        return False
    norm = text.lower()
    norm = _NBSP_RE.sub(" ", norm)
    norm = re.sub(r"\s+", " ", norm)
    norm = norm.strip()
    if not norm:
        return False
    collapsed = re.sub(r"\s+", "", norm)
    for neg in _GROSS_NEG_PATTERNS:
        if neg.search(norm) or neg.search(collapsed):
            return False
    for pos in _GROSS_POS_PATTERNS:
        if pos.search(norm) or pos.search(collapsed):
            return True
    # дополнительная проверка на маркеры без пробелов
    for marker in ("довычета", "довычетаналога", "додержания", "додержанияндфл", "доуплатыналогов"):
        if marker in collapsed:
            return True
    return False

# ================= РОЛЕВЫЕ ФИЛЬТРЫ ПО НАЗВАНИЮ =================
FILTERS = {
    "повар": {
        "inc": [r"\bповар\b", r"\bшеф-?повар\b", r"\bсу-?шеф\b",
                r"\bпиццамейкер\b", r"\bсушист\b", r"\bкондитер\b", r"\bпекар\b"],
        "exc": [r"\bаттракци", r"\bкол-?центр\b", r"\bcall[- ]?центр\b",
                r"\bпродаж", r"\bкассир\b", r"\bстанк", r"\bазс\b", r"\bзаправк"]
    },
    "повар_холодного_цеха": {
        "inc": [r"\bповар\b.*\bхолодн\w*\b", r"\bповар холодного цеха\b", r"\bхолодный цех\b"],
        "exc": [r"\bаттракци", r"\bкассир\b", r"\bстанк", r"\bcall[- ]?центр\b", r"\bофициант\b", r"\bбармен\b"]
    },
    "повар_горячего_цеха": {
        "inc": [r"\bповар\b.*\bгоряч\w*\b", r"\bповар горячего цеха\b", r"\bгорячий цех\b"],
        "exc": [r"\bаттракци", r"\bкассир\b", r"\bстанк", r"\bcall[- ]?центр\b", r"\bофициант\b", r"\bбармен\b"]
    },
    "оператор_доставки": {
        "inc": [
            r"\bоператор\W{0,3}достав", r"\bоператор заказов\b",
            r"\bдиспетчер[- ]?достав", r"\bкоординатор достав",
            r"\bоператор (?:пвз|пункта выдачи)\b", r"\bпункт выдачи\b",
            r"\bсборщик заказов\b"
        ],
        "exc": [
            r"\bаттракци", r"\bкассир\b", r"\bпродаж", r"\bстанк",
            r"\bcall[- ]?центр\b", r"\bофициант\b", r"\bбармен\b"
        ]
    },
    "кассир": {
        "inc": [
            r"\bкассир\b", r"\bстарший\s+кассир\b",
            r"\bкассир[- ]операционист\b", r"\bпродавец[- ]кассир\b",
            r"\bкассир[- ]консультант\b", r"\bкассир[- ]смены\b",
        ],
        "exc": [
            r"\bбариста\b", r"\bадминистратор\b", r"\bбухгалтер\w*\b",
            r"\bоператор\b", r"\bcall[- ]?центр\b",
            r"\bофициант\b", r"\bбармен\b", r"\bповар\b",
            r"^(?=.*продавец[- ]?консультант)(?!.*кассир).*$",
        ],
    },
    "менеджер_разработки_продукта": {
        "inc": [
            r"\bменеджер\b.*\bразработк\w*\b.*\bпродукт",
            r"\bменеджер\b.*\bразработк\w*\b.*\bменю\b",
            r"\bменеджер\b.*\bблюд\w*\b", r"\bпродукта\b",
            r"\bменеджер\s*r\W?&\W?d\b",
            r"\bproduct\s*development\b",
        ],
        "exc": [
            r"\bproduct\s*manager\b",
            r"\bIT\b|\bайти\b|\bdigital\b|\bsoftware\b|\bприложен|\bПО\b",
        ],
    },
    "бариста": {
        "inc": [
            r"\bбариста\b",
            r"\bстарший\s+бариста\b",
            r"\bbarista\b",
            r"\bкофе[йи]\w*\s*мастер\b",
        ],
        "exc": [
            r"\bофициант\b",
            r"\bповар\b",
            r"\bбармен\b",
            r"\bкассир\b",
            r"\bадминистратор\b",
            r"\bоператор\b",
            r"\bcall[- ]?центр\b",
            r"\bпромоутер\b",
            r"\bкурьер\b",
            r"\bпродаж",
            r"\bаттракци",
        ],
    },
    "smm_менеджер": {
        "inc": [
            r"\bsmm\b",
            r"\bsmm[- ]?менеджер\b",
            r"\bsmm[- ]?manager\b",
            r"\bспециалист\s+по\s+smm\b",
            r"\bsmm[- ]?специалист\b",
            r"\bsmm[- ]?маркетолог\b",
        ],
        "exc": [
            r"\bcall[- ]?центр\b", r"\bоператор\b", r"\bпродаж", r"\bsales\b",
            r"\bseo\b", r"\bppc\b", r"\bконтекст\b", r"\bтаргетолог\b",
            r"\bкопирайтер\b", r"\bдизайнер\b", r"\bmotion\b", r"\bвидеограф\b",
            r"\bpr\b|\bpublic\s*relations\b", r"\bаккаунт[- ]?менеджер\b",
            r"\bадминистратор\b", r"\bофициант\b", r"\bкурьер\b"
        ],
    }
}

def _compile_filters(role: str):
    cfg = FILTERS.get(role, {})
    inc = cfg.get("inc", [r".*"])
    exc = cfg.get("exc", [])
    INC = re.compile("|".join(inc), re.I)
    EXC = re.compile("|".join(exc), re.I) if exc else None
    return INC, EXC

def keep_by_title(title: str, INC, EXC) -> bool:
    t = title or ""
    if EXC and EXC.search(t): return False
    return bool(INC.search(t))

# ================= УТИЛИТЫ ПАРСИНГА ТЕКСТА =================
def _strip_html(s: Optional[str]) -> str:
    if not s: return ""
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def extract_comp(text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Возвращает (hour, shift12). Если найдена почасовая — смена 12ч считается всегда.
    Дополнительно ловим: "₽/ч", "р/ч", "ставка ... ₽/час", "за смену N".
    """
    t = (text or "").lower().replace("\xa0", " ")
    def to_num(s: str) -> Optional[float]:
        return float(re.sub(r"[^\d]", "", s)) if s and re.search(r"\d", s) else None

    # 1) Прямые указания почасовой ставки
    m = re.search(r"(?:^|[^\d])(\d[\d\s]{2,})\s*(?:₽|р|руб)\s*/\s*(?:ч|час)\b", t, re.I) \
        or re.search(r"\bставк\w*\s*(\d[\d\s]{2,})\s*(?:₽|р|руб)\s*/?\s*(?:ч|час)\b", t, re.I)
    if m:
        hour = to_num(m.group(1))
        return (round(hour, 2) if hour else None, (hour * 12.0 if hour else None))

    # 2) "за смену N"
    m = re.search(r"\bза\s*смену\s*(\d[\d\s]{3,})\s*(?:₽|р|руб)\b", t, re.I)
    if m:
        shift = to_num(m.group(1))
        return (round(shift / 12.0, 2) if shift else None, round(shift, 2) if shift else None)

    # 3) Старые паттерны
    m_hour  = re.search(r"(\d[\d\s]{2,})\s*(?:₽|руб)\s*(?:/|за)?\s*час", t, re.I)
    m_shift = re.search(r"(\d[\d\s]{3,})\s*(?:₽|руб).{0,25}(?:смен[аы]|12\s*час)", t, re.I)
    hour  = to_num(m_hour.group(1)) if m_hour else None
    shift = to_num(m_shift.group(1)) if m_shift else None

    # защита от "руб/мес"
    if hour and m_hour:
        span = m_hour.span()
        win = t[max(0, span[0]-20):min(len(t), span[1]+20)]
        if re.search(r"(мес|месяц|год)", win):
            hour = None

    if hour and not shift:
        shift = hour * 12.0
    if shift and not hour:
        hour = shift / 12.0
    return (round(hour,2) if hour else None, round(shift,2) if shift else None)


DEFAULT_SHIFT_HOURS = 12.0
SHIFTS_PER_MONTH_2X2 = 15.0
_MONTHLY_THRESHOLD_TR = 15.0  # >=15 т.р. считаем месячной ставкой


def _avg_salary_thousand(*values: Optional[float]) -> Optional[float]:
    vals = []
    for raw in values:
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val <= 0:
            continue
        vals.append(val)
    if not vals:
        return None
    return sum(vals) / len(vals)


def derive_hour_shift_from_salary(
    sal_from_tr: Optional[float],
    sal_to_tr: Optional[float],
    shift_hours: Optional[float],
) -> Tuple[Optional[float], Optional[float], float]:
    """Оценить часовую ставку и оплату смены по ЗП."""

    eff_hours = shift_hours if isinstance(shift_hours, (int, float)) and shift_hours > 0 else DEFAULT_SHIFT_HOURS
    avg_thousand = _avg_salary_thousand(sal_from_tr, sal_to_tr)
    if avg_thousand is None:
        return None, None, eff_hours

    avg_rub = avg_thousand * 1000.0
    denom = eff_hours
    if avg_thousand >= _MONTHLY_THRESHOLD_TR:
        denom *= SHIFTS_PER_MONTH_2X2
    if denom <= 0:
        return None, None, eff_hours

    hour = avg_rub / denom
    if hour <= 0:
        return None, None, eff_hours

    hour = round(hour, 2)
    shift_val = round(hour * eff_hours, 2)
    return hour, shift_val, eff_hours

NUM_WORD = {"сутки":1,"день":1,"один":1,"одна":1,"два":2,"две":2,"три":3,"четыре":4,"пять":5,"шесть":6,"семь":7}
SCHED_NUM_RE = re.compile(r"\b([1-9]\d?)\s*[/\-–xх×]\s*([1-9]\d?)\b", re.I)
def _words_pair(t: str) -> Optional[str]:
    m = re.search(rf"\b({'|'.join(NUM_WORD)})\s+через\s+({'|'.join(NUM_WORD)})\b", t, re.I)
    if not m: return None
    a = NUM_WORD.get(m.group(1).lower()); b = NUM_WORD.get(m.group(2).lower())
    if a and b: return f"{a}/{b}"
    return None

def extract_schedule_strict(text: str, sched_src: Optional[str]=None) -> Optional[str]:
    t = (text or "") + " " + (sched_src or "")
    t = t.lower().replace("–","-").replace("х","x")
    vals = []
    for m in SCHED_NUM_RE.finditer(t):
        vals.append(f"{int(m.group(1))}/{int(m.group(2))}")
    wp = _words_pair(t)
    if wp: vals.append(wp)
    for m in re.finditer(r"\bвахт\w*\s*([1-9]\d?)\s*[/\-–xх×]\s*([1-9]\d?)\b", t, re.I):
        vals.append(f"{int(m.group(1))}/{int(m.group(2))}")
    out, seen = [], set()
    for v in vals:
        if v not in seen:
            seen.add(v); out.append(v)
    return ", ".join(out) if out else None

def _extract_schedule_from_html(url: str, timeout: float = 15.0) -> Tuple[Optional[str], Optional[float]]:
    if not (_HAS_BS4 and isinstance(url,str) and url.startswith("http")):
        return None, None
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code != 200:
            return None, None
        soup = BeautifulSoup(r.text, "lxml")
        txts = []
        for sel in [
            "[data-qa='vacancy-view-employment-mode']",
            "[data-qa='vacancy-view-raw__schedule']",
            "[data-qa='vacancy-view-raw__workingschedule']",
            "[data-qa='vacancy-view-employment-type']",
            "[data-qa='vacancy-view-raw__main-info']",
            "[data-qa='vacancy-view-employment-mode-item']",
        ]:
            for el in soup.select(sel):
                txts.append(el.get_text(" ", strip=True))
        for label in soup.find_all(string=re.compile(r"(График|График работы|Рабочие часы|Смена)", re.I)):
            parent = getattr(label, "parent", None)
            s = " ".join(parent.stripped_strings) if parent else str(label)
            txts.append(s)
        blob = " | ".join(txts).lower()
        blob = html.unescape(blob).replace("–","-").replace("х","x")
        graph = None
        m = SCHED_NUM_RE.search(blob)
        if m: graph = f"{int(m.group(1))}/{int(m.group(2))}"
        if not graph:
            graph = _words_pair(blob)
        mh = re.search(r"(?:длительность|рабочие\s*часы|смена)\D{0,12}(\d{1,2})\s*час", blob)
        hours = float(mh.group(1)) if mh else None
        return graph, hours
    except Exception:
        return None, None

def extract_pay_frequency(text: str) -> Optional[str]:
    if not text: return None
    t = text.lower()
    if re.search(r"еженедел|каждую неделю|раз в неделю|weekly", t): return "Еженедельно"
    if re.search(r"2 раза в месяц|два раза в месяц|аванс", t):     return "2 раза в месяц"
    if re.search(r"ежемесяч|раз в месяц|monthly", t):               return "Ежемесячно"
    return None

def extract_employment_type(text: str, employment_name: Optional[str] = None) -> Optional[str]:
    t = (text or "").lower(); e = (employment_name or "").lower()
    if re.search(r"гпх|гражданско-правов|самозанят|подряд|аутстаф", t): return "ГПХ"
    if re.search(r"по тк|трудов|официальн|оформление по тк|белая зп", t): return "ТК"
    if re.search(r"полная|частичная|полный|частичный", e): return "ТК"
    return None

def extract_experience(text: str) -> Optional[str]:
    t = (text or "").lower()
    if not t:
        return None
    if "без опыта" in t:
        return "Без опыта"

    # полгода / полугода
    if re.search(r"опыт(?: работы)?\s*(?:от\s*)?(полгода|пол\s*года)", t):
        return "от 0.5 года"

    patterns = [
        r"опыт(?: работы)?\s*(?:не менее\s*)?(?:от\s*)?(\d+(?:[\.,]\d+)?)\s*(год|лет|года|месяц|месяца|месяцев)",
        r"от\s*(\d+(?:[\.,]\d+)?)\s*(год|лет|года|месяц|месяца|месяцев)\s*(?:опыта|стажа)",
        r"(\d+(?:[\.,]\d+)?)\s*(год|лет|года|месяц|месяца|месяцев)\s*опыт"
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if not m:
            continue
        raw_val, unit = m.group(1), m.group(2)
        val = raw_val.replace(",", ".")
        try:
            num = float(val)
        except ValueError:
            num = None

        if num is not None and abs(num - round(num)) < 1e-6:
            num = int(round(num))
        value_str = str(num if num is not None else raw_val).strip()

        unit = unit.lower()
        if unit.startswith("месяц"):
            unit = "месяцев" if value_str not in {"1", "1.0"} else "месяц"
        else:
            # год/лет склонение
            if value_str in {"1", "1.0"}:
                unit = "год"
            elif value_str in {"2", "3", "4"}:
                unit = "года"
            else:
                unit = "лет"

        segment = m.group(0)
        prefix = "от " if re.search(r"от|не менее", segment) else ""
        return f"{prefix}{value_str} {unit}".strip()

    if "опыт" in t:
        return "Опыт требуется"
    return None

SECTION_HEADS = ["обязанности","что делать","чем предстоит заниматься","задачи"]
NEXT_HEADS = ["требования","условия","мы предлагаем","о компании","график","контакты","оформление","что мы предлагаем"]
def extract_responsibilities(html_or_text: str, fallback: Optional[str] = None) -> Optional[str]:
    text = _strip_html(html_or_text); low = text.lower()
    start = None
    for h in SECTION_HEADS:
        for sep in (":"," :","\n"):
            i = low.find(h + sep)
            if i != -1:
                start = i + len(h) + len(sep); break
        if start is not None: break
    if start is None:
        lines = [l.strip(" -•—\t") for l in text.splitlines() if l.strip().startswith(("—","-","•"))]
        return ("; ".join([l for l in lines if l])[:3000] or fallback or (text[:3000] if text else None))
    tail = text[start:]; end = len(tail); low_tail = tail.lower()
    for nh in NEXT_HEADS:
        for sep in (":","\n"):
            j = low_tail.find(nh + sep)
            if j != -1: end = min(end, j)
    body = tail[:end]
    lines = [re.sub(r"^[\s\-•—]+", "", l).strip() for l in body.splitlines()]
    lines = [l for l in lines if l]
    return ("; ".join(lines)[:3000]) if lines else fallback

def extract_shift_len(text: str) -> Optional[tuple]:
    t = (text or "").lower().replace("\xa0", " ").replace("–", "-")
    m = re.search(r"\bпо\s*(\d{1,2})\s*[-–]\s*(\d{1,2})\s*час", t) or \
        re.search(r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\s*час", t) or \
        re.search(r"\bс\s*(\d{1,2})\s*до\s*(\d{1,2})\s*час", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if 1 <= a <= 24 and 1 <= b <= 24:
            return ("text", f"'{a}-{b}")
    m = re.search(r"\b(\d{1,2})\s*[- ]?\s*час(?:овая)?\b|\bсмена\s*(\d{1,2})\s*час", t)
    if m:
        v = m.group(1) or m.group(2)
        h = float(v)
        if 1 <= h <= 24:
            return ("num", h)
    m = re.search(r"\b(?:\d\s*/\s*\d|сутк\w*/\w+)\b.*?\b(\d{1,2})\s*час", t)
    if m:
        h = float(m.group(1))
        if 1 <= h <= 24:
            return ("num", h)
    return None

BENEFITS = ["дмс","медицинская страховка","страхование","питание","бесплатное питание","корпоративное питание",
            "форма","униформа","спецодежда","премии","бонус","бонусы","подарки","скидки","обучение",
            "проезд","оплата проезда","жилье","жильё","общежитие","развозка","транспорт","кофе","чай"]
def pick_benefits(text: str) -> Optional[str]:
    t = (text or "").lower()
    out, seen = [], set()
    for kw in BENEFITS:
        if kw in t and kw not in seen:
            seen.add(kw); out.append(kw.upper() if kw=="дмс" else kw)
    return ", ".join(out) if out else None

# =================== HH.RU ===================
def _iso_date(s: Optional[str]) -> Optional[str]:
    if not s: return None
    try:
        if len(s) >= 5 and (s[-5] in "+-") and s[-3] != ":":
            s = s[:-2] + ":" + s[-2:]
        dt = datetime.fromisoformat(s.replace("Z","+00:00"))
        return dt.date().isoformat()
    except Exception:
        return s.split("T",1)[0]

def hh_search(query: str, area: int, pages: int, per_page: int, pause: float, search_in: str) -> List[Dict[str, Any]]:
    sess = _get_sess()
    items = []
    for page in range(pages):
        p = {"text": query, "area": area, "page": page, "per_page": per_page, "only_with_salary": "false"}
        if search_in in ("name","description","company_name","everything"):
            p["search_field"] = search_in
        try:
            r = sess.get("https://api.hh.ru/vacancies", params=p, timeout=_TIMEOUT)
            if r.status_code != 200:
                break
            data = r.json()
            items += data.get("items", [])
            if page >= data.get("pages", 0) - 1:
                break
        except requests.RequestException as e:
            print(f"[HH] search fail (page={page}): {e.__class__.__name__}")
            break
        time.sleep(pause)
    return items

def hh_details(vac_id: str) -> dict:
    sess = _get_sess()
    try:
        r = sess.get(f"https://api.hh.ru/vacancies/{vac_id}", timeout=_TIMEOUT)
        if r.status_code == 200:
            return r.json()
        print(f"[HH] details non-200 for {vac_id}: {r.status_code}")
    except requests.RequestException as e:
        print(f"[HH] details fail {vac_id}: {e.__class__.__name__}")
    return {}

def map_hh(items: List[Dict[str, Any]], pause_detail: float = 0.2) -> List[Dict[str, Any]]:
    rows = []
    for v in items:
        vid = v.get("id")
        name = v.get("name")
        employer = (v.get("employer") or {}).get("name")
        url = v.get("alternate_url") or v.get("url")
        salary = v.get("salary") or {}
        cur = salary.get("currency") or "RUR"
        to_tr = lambda x: round(x / 1000.0, 1) if (x is not None and x > 0 and cur == "RUR") else None
        sal_from_tr = to_tr(salary.get("from"))
        sal_to_tr   = to_tr(salary.get("to"))
        exp = (v.get("experience") or {}).get("name")
        empl_src = (v.get("employment") or {}).get("name")
        sched_src = (v.get("schedule") or {}).get("name")
        snip = v.get("snippet") or {}
        resp_snip = snip.get("responsibility") or ""
        reqs_snip = snip.get("requirement") or ""
        short = f"{resp_snip} {reqs_snip}"
        det = hh_details(vid) if vid else {}
        descr_html = det.get("description") or ""
        descr_txt = _strip_html(descr_html) or short
        published = _iso_date(v.get("published_at") or det.get("published_at"))
        hour, shift = extract_comp(descr_txt)
        graph = extract_schedule_strict(descr_txt, sched_src=None)
        sl = extract_shift_len(descr_txt)
        if sl:
            shift_len = sl[1]
            if isinstance(shift_len, (int, float)) and hour and not shift:
                shift = hour * float(shift_len)
        else:
            shift_len = 12.0 if (hour or shift) else None
        if (not graph or shift_len is None) and isinstance(url, str) and url.startswith("http"):
            g_html, hours_html = _extract_schedule_from_html(url)
            if not graph and g_html:
                graph = g_html
            if shift_len is None and hours_html:
                shift_len = float(hours_html)
                if hour and not shift:
                    shift = hour * float(shift_len)
        shift_hours_val = shift_len if isinstance(shift_len, (int, float)) else None
        if hour is None:
            salary_hour, salary_shift, eff_hours = derive_hour_shift_from_salary(sal_from_tr, sal_to_tr, shift_hours_val)
            if salary_hour is not None:
                hour = salary_hour
                if shift is None and salary_shift is not None:
                    shift = salary_shift
                if shift_len is None:
                    shift_len = eff_hours

        shift12_out = shift if (isinstance(shift_len, (int, float)) and float(shift_len) == 12.0) else (hour * 12.0 if hour else None)
        pay    = extract_pay_frequency(descr_txt)
        employ = extract_employment_type(descr_txt, employment_name=empl_src)
        duties = extract_responsibilities(descr_html or descr_txt, fallback=resp_snip or reqs_snip)
        bens   = pick_benefits(descr_txt)
        gross_text = _gross_text_blob(
            name,
            salary,
            det.get("salary") if isinstance(det, dict) else None,
            det.get("compensation") if isinstance(det, dict) else None,
            resp_snip,
            reqs_snip,
            short,
            descr_txt,
            duties,
        )
        gross_note = GROSS_NOTE_TEXT if is_gross_salary(gross_text) else ""
        rate_text = _rate_text_blob(
            name,
            salary,
            det.get("salary") if isinstance(det, dict) else None,
            det.get("compensation") if isinstance(det, dict) else None,
            resp_snip,
            reqs_snip,
            short,
            descr_txt,
            duties,
            bens,
            graph,
        )

        rows.append({
            "Должность": name,
            "Работодатель": employer,
            "Дата публикации": published,
            "ЗП от (т.р.)": sal_from_tr,
            "ЗП до (т.р.)": sal_to_tr,
            "Средний совокупный доход при графике 2/2 по 12 часов": shift12_out,
            "В час": hour,
            "Длительность \nсмены": shift_len,
            "Требуемый\nопыт": exp or None,
            "Труд-во": employ,
            "График": graph,
            "Частота \nвыплат": pay,
            "Льготы": bens,
            "Обязаности": duties,
            "Ссылка": url,
            "Примечание": gross_note,
            "gross_note": gross_note,
            "__text": f"{descr_txt} {duties or ''}",
            "__rate_text": rate_text,
        })
        time.sleep(pause_detail)
    return rows

# --- Город Работ: нормализация карточек в формат Excel ---
def map_gr(rows_in):
    import requests, json, re
    from urllib.parse import urlparse
    try:
        from bs4 import BeautifulSoup
    except Exception:
        BeautifulSoup = None

    sess = _mk_polite_session()

    # собрать уникальные URL из gr_search
    urls, seen = [], set()
    titles_prefill = {}
    text_prefill = {}
    for r in rows_in or []:
        u = (r.get("url") or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        urls.append(u)
        t = (r.get("title") or "").strip()
        if t:
            titles_prefill[u] = t
        blob = _gross_text_blob(r)
        if blob:
            text_prefill[u] = blob

    if not urls:
        return []

    def _first_jobposting(soup):
        # вернуть первый JSON-LD типа JobPosting, если есть
        for tag in soup.find_all("script", {"type": "application/ld+json"}):
            try:
                data = json.loads(tag.string)
                arr = data if isinstance(data, list) else [data]
                for obj in arr:
                    if isinstance(obj, dict) and obj.get("@type") in ("JobPosting", ["JobPosting"]):
                        return obj
            except Exception:
                continue
        return None

    def _one(u):
        host = urlparse(u).netloc.lower()
        title = titles_prefill.get(u)
        emp = desc = None
        pub = None; sal_from = sal_to = None
        jp = None

        soup = None
        try:
            r = _polite_get(sess, u, timeout=15, site="gr")
            if r is not None and BeautifulSoup and r.text:
                soup = BeautifulSoup(r.text, "lxml")
            if r is not None and r.status_code != 200 and not soup:
                return {
                    "Должность": title or "Вакансия",
                    "Работодатель": None,
                    "Дата публикации": None,
                    "ЗП от (т.р.)": None,
                    "ЗП до (т.р.)": None,
                    "Средний совокупный доход при графике 2/2 по 12 часов": None,
                    "В час": None,
                    "Длительность \nсмены": None,
                    "Требуемый\nопыт": None,
                    "Труд-во": None,
                    "График": None,
                    "Частота \nвыплат": None,
                    "Льготы": None,
                    "Обязаности": None,
                    "Ссылка": u,
                    "Примечание": "",
                    "gross_note": "",
                    "__text": None,
                }
        except Exception:
            soup = None

        # 1) Если это именно gorodrabot.ru — пробуем JSON-LD
        if soup is not None and "gorodrabot.ru" in host:
            jp = _first_jobposting(soup) or {}
            if jp:
                title = title or jp.get("title") or jp.get("name")
                desc  = jp.get("description") or desc
                org   = jp.get("hiringOrganization") or {}
                if isinstance(org, dict):
                    emp = org.get("name") or emp
                date_iso = jp.get("datePosted")
                if date_iso:
                    pub = (date_iso.split("T", 1)[0]).split("+", 1)[0]
                sal = jp.get("baseSalary") or {}
                try:
                    val = sal.get("value") if isinstance(sal, dict) else None
                    if isinstance(val, dict):
                        vmin = val.get("minValue") or val.get("value")
                        vmax = val.get("maxValue")
                        if vmin: sal_from = round(float(vmin) / 1000.0, 1)
                        if vmax: sal_to   = round(float(vmax) / 1000.0, 1)
                except Exception:
                    pass

        # 2) Фоллбэк по мета-тегам/титлу (для любых доменов)
        if soup is not None:
            if not title:
                try:
                    og = soup.find("meta", {"property": "og:title"})
                    if og and og.get("content"):
                        title = og["content"].strip()
                except Exception:
                    pass
            if not title and soup.title and soup.title.text:
                title = soup.title.text.strip()

            if not emp:
                try:
                    emp_el = soup.select_one('[itemprop="hiringOrganization"], [data-qa="vacancy-company-name"]')
                    if emp_el:
                        emp = emp_el.get_text(" ", strip=True)
                except Exception:
                    pass
            if not desc:
                try:
                    de = soup.select_one('[itemprop="description"], .description, [data-qa="vacancy-view-description"]')
                    if de:
                        desc = de.get_text(" ", strip=True)
                except Exception:
                    pass

        descr = (desc or "").strip()
        hour, shift_sum = extract_comp(descr)
        shift_len = None
        sl = extract_shift_len(descr)
        if sl:
            shift_len = sl[1]
            if isinstance(shift_len, (int, float)) and hour and not shift_sum:
                shift_sum = hour * float(shift_len)
        else:
            shift_len = 12.0 if (hour or shift_sum) else None
        shift_hours_val = shift_len if isinstance(shift_len, (int, float)) else None
        if hour is None:
            salary_hour, salary_shift, eff_hours = derive_hour_shift_from_salary(sal_from, sal_to, shift_hours_val)
            if salary_hour is not None:
                hour = salary_hour
                if shift_sum is None and salary_shift is not None:
                    shift_sum = salary_shift
                if shift_len is None:
                    shift_len = eff_hours
        shift12_out = shift_sum if (isinstance(shift_len, (int, float)) and float(shift_len) == 12.0) else (hour * 12.0 if hour else None)

        graph = extract_schedule_strict(descr, sched_src=None)
        pay = extract_pay_frequency(descr)
        employ = extract_employment_type(descr)
        duties = extract_responsibilities(desc or "", fallback=(descr[:3000] if descr else None))
        bens = pick_benefits(descr)
        exp = extract_experience(descr)

        gross_text = _gross_text_blob(
            title,
            emp,
            descr,
            duties,
            text_prefill.get(u),
            jp,
        )
        gross_note = GROSS_NOTE_TEXT if is_gross_salary(gross_text) else ""
        rate_text = _rate_text_blob(
            title,
            emp,
            descr,
            duties,
            bens,
            text_prefill.get(u),
            jp,
            graph,
        )

        return {
            "Должность": (title or "Вакансия").strip(),
            "Работодатель": emp,
            "Дата публикации": pub,
            "ЗП от (т.р.)": sal_from,
            "ЗП до (т.р.)": sal_to,
            "Средний совокупный доход при графике 2/2 по 12 часов": shift12_out,
            "В час": hour,
            "Длительность \nсмены": shift_len,
            "Требуемый\nопыт": exp,
            "Труд-во": employ,
            "График": graph,
            "Частота \nвыплат": pay,
            "Льготы": bens,
            "Обязаности": duties,
            "Ссылка": u,
            "Примечание": gross_note,
            "gross_note": gross_note,
            "__text": f"{descr} {duties or ''}".strip() or None,
            "__rate_text": rate_text,
        }

    out = []
    # параллелизм можно оставить/убрать; если были проблемы — ставь max_workers=4
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=min(_WORKERS, 4)) as ex:
        futs = [ex.submit(_one, u) for u in urls]
        for f in as_completed(futs):
            try:
                row = f.result()
                if row:
                    out.append(row)
            except Exception:
                pass
    return out

# =================== gorodrabot.ru ===================
def _text(node) -> str:
    return re.sub(r"\s+"," ", node.get_text(strip=True)) if node else ""


def _rub_to_tr(s: Optional[str]) -> Tuple[Optional[float],Optional[float]]:
    if not s: return None, None
    sums = re.findall(r"(\d[\d\s]{3,})\s*(?:₽|руб)", s.lower())
    vals=[]
    for part in sums:
        v = int(re.sub(r"\D","", part))
        if 1000 <= v <= 10_000_000:
            vals.append(v)
    if not vals: return None, None
    return round(min(vals)/1000.0,1), round(max(vals)/1000.0,1)

def _ensure_city_slug(city: str) -> str:
    m = {
        "москва": "moskva", "moscow": "moskva",
        "санкт-петербург": "sankt-peterburg", "spb": "sankt-peterburg",
        "казань": "kazan", "новосибирск": "novosibirsk"
    }
    s = (city or "").strip().lower()
    return m.get(s, "rossiya")  # <-- запасной общий регион



def _parse_rel_date_avito(text: str) -> Optional[str]:
    if not text: return None
    t = text.lower().strip()
    now = datetime.now(timezone(timedelta(hours=3)))
    if "сегодня" in t: return now.date().isoformat()
    if "вчера" in t:   return (now - timedelta(days=1)).date().isoformat()
    m = re.search(r"(\d+)\s*(минут|час|часа|часов|день|дня|дней|недел)", t)
    if m:
        n = int(m.group(1)); u = m.group(2)
        delta = {"минут":("minutes",n),"час":("hours",n),"часа":("hours",n),"часов":("hours",n),
                 "день":("days",n),"дня":("days",n),"дней":("days",n),"недел":("weeks",n)}
        kind,val = delta.get(u,("days",0))
        return (now - timedelta(**{kind:val})).date().isoformat()
    m2 = re.search(r"(\d{1,2})\s+([а-я]+)", t)
    if m2:
        day = int(m2.group(1)); mon = m2.group(2)[:5]
        mon_map = {"январ":1,"феврал":2,"март":3,"апрел":4,"ма":5,"июн":6,"июл":7,"август":8,"сентябр":9,"октябр":10,"ноябр":11,"декабр":12}
        for k, mm in mon_map.items():
            if mon.startswith(k):
                year = now.year - (mm > now.month)
                try:
                    return datetime(year, mm, day, tzinfo=timezone(timedelta(hours=3))).date().isoformat()
                except ValueError:
                    return None
    return None

# =================== Avito ===================

ITEM_URL_RE = re.compile(r"^https?://(?:www\.)?avito\.ru/.*/vakansii/[^/?#]+-\d{6,}(?:[/?#].*)?$")
def _is_avito_item_url(u: str) -> bool:
    return bool(ITEM_URL_RE.match((u or "").strip().lower()))

def avito_search(query: str, city: str, pages: int, pause: float,
                 headful=False, state_path="avito_state.json") -> List[Dict[str, Any]]:
    city_slug = _ensure_city_slug(city)
    urls = _avito_listing_desktop(query, city_slug, pages, pause, headful=headful, state_path=state_path)
    return [{"title": "", "price": "", "date": None, "url": u} for u in urls]



def _save_debug(page, name: str):
    if not AVITO_DEBUG:
        return
    dbg = (EXPORT_DIR / "_debug")
    dbg.mkdir(parents=True, exist_ok=True)
    try:
        page.screenshot(path=str(dbg / f"{name}.png"), full_page=True)
    except Exception:
        pass
    try:
        html = page.content() or ""
        (dbg / f"{name}.html").write_text(html, encoding="utf-8")
    except Exception:
        pass

GR_DEBUG = True  # можно потом выключить

def _gr_save_debug(html_text: str, name: str):
    if not GR_DEBUG:
        return
    dbg = EXPORT_DIR / "_debug"
    dbg.mkdir(parents=True, exist_ok=True)
    try:
        (dbg / f"{name}.html").write_text(html_text or "", encoding="utf-8")
    except Exception:
        pass


def map_avito(rows_in: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    urls = []
    seen = set()
    prefill_text: dict[str, str] = {}
    for r in rows_in:
        raw_url = r.get("url") or r.get("Ссылка") or ""
        u = _normalize_url(raw_url)
        blob = _gross_text_blob(r)
        if blob and u:
            prefill_text[u] = blob
        if _is_item(u) and u not in seen:
            seen.add(u); urls.append(u)
    if not urls:
        return []

    import requests
    sess = _mk_polite_session()

    def _first_jobposting(soup):
        for tag in soup.find_all("script", {"type": "application/ld+json"}):
            try:
                data = json.loads(tag.string)
                arr = data if isinstance(data, list) else [data]
                for obj in arr:
                    if isinstance(obj, dict) and obj.get("@type") in ("JobPosting", ["JobPosting"]):
                        return obj
            except Exception:
                continue
        return None

    mapped = []
    def grab(u: str) -> Dict[str, Any]:
        title = emp = desc = None
        pub = None; sal_from = sal_to = None
        soup = None
        jp = None
        try:
            d = _polite_get(sess, u, _TIMEOUT, site="avito")
            if (not d) or (d.status_code != 200) or ("<title>" not in d.text):
                d = _polite_get(sess, _to_mobile(u), _TIMEOUT, site="avito")
            if d.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(d.text, "lxml")
        except Exception:
            soup = None

        if soup is not None:
            jp = _first_jobposting(soup) or {}
            if jp:
                title = jp.get("title") or jp.get("name") or title
                desc  = jp.get("description") or desc
                org   = jp.get("hiringOrganization") or {}
                if isinstance(org, dict): emp = org.get("name") or emp
                date_iso = jp.get("datePosted")
                if date_iso: pub = (date_iso.split("T",1)[0]).split("+",1)[0]
                sal = jp.get("baseSalary") or {}
                try:
                    val = sal.get("value") if isinstance(sal, dict) else None
                    if isinstance(val, dict):
                        vmin = val.get("minValue") or val.get("value")
                        vmax = val.get("maxValue")
                        if vmin: sal_from = round(float(vmin)/1000.0, 1)
                        if vmax: sal_to   = round(float(vmax)/1000.0, 1)
                except Exception:
                    pass
            if not title:
                og = soup.find("meta", {"property": "og:title"})
                if og and og.get("content"): title = og["content"]
            if not title and soup.title and soup.title.text:
                title = soup.title.text.strip()
            if not emp:
                el = soup.select_one('[data-marker="seller-info/name"], [data-marker="seller-link/linkText"]')
                emp = el.get_text(strip=True) if el else None
            if not desc:
                el = soup.select_one('[data-marker="item-description/text"], [itemprop="description"]')
                desc = el.get_text(" ", strip=True) if el else None

        # derive
        descr = (desc or "").strip()
        hour, shift_sum = extract_comp(descr)
        shift_len = None
        sl = extract_shift_len(descr)
        if sl:
            shift_len = sl[1]
            if isinstance(shift_len, (int, float)) and hour and not shift_sum:
                shift_sum = hour * float(shift_len)
        else:
            shift_len = 12.0 if (hour or shift_sum) else None
        shift_hours_val = shift_len if isinstance(shift_len, (int, float)) else None
        if hour is None:
            salary_hour, salary_shift, eff_hours = derive_hour_shift_from_salary(sal_from, sal_to, shift_hours_val)
            if salary_hour is not None:
                hour = salary_hour
                if shift_sum is None and salary_shift is not None:
                    shift_sum = salary_shift
                if shift_len is None:
                    shift_len = eff_hours
        shift12_out = shift_sum if (isinstance(shift_len, (int, float)) and float(shift_len) == 12.0) else (hour * 12.0 if hour else None)

        graph = extract_schedule_strict(descr, sched_src=None)
        pay   = extract_pay_frequency(descr)
        employ= extract_employment_type(descr)
        duties= extract_responsibilities(desc or "", fallback=(descr[:3000] if descr else None))
        bens  = pick_benefits(descr)
        exp   = extract_experience(descr)

        gross_text = _gross_text_blob(
            title,
            emp,
            desc,
            duties,
            prefill_text.get(u),
            jp,
        )
        gross_note = GROSS_NOTE_TEXT if is_gross_salary(gross_text) else ""

        return {
            "Должность": (title or "Вакансия").strip(),
            "Работодатель": emp,
            "Дата публикации": pub,
            "ЗП от (т.р.)": sal_from,
            "ЗП до (т.р.)": sal_to,
            "Средний совокупный доход при графике 2/2 по 12 часов": shift12_out,
            "В час": hour,
            "Длительность \nсмены": shift_len,
            "Требуемый\nопыт": exp,
            "Труд-во": employ,
            "График": graph,
            "Частота \nвыплат": pay,
            "Льготы": bens,
            "Обязаности": duties,
            "Ссылка": u,
            "Примечание": gross_note,
            "gross_note": gross_note,
            "__text": f"{descr} {duties or ''}".strip() or None,
        }

    with ThreadPoolExecutor(max_workers=min(_WORKERS, 3)) as ex:
        for f in as_completed([ex.submit(grab, u) for u in urls]):
            try:
                row = f.result()
                if row: mapped.append(row)
            except Exception:
                continue
    return mapped
# --------------------- Город Работ (gorodrabot.ru) ---------------------
def gr_search(query: str, city: str, pages: int, pause: float,
              headful: bool = False, state_path: str | None = "gr_state.json") -> list[dict]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        print("[GR] Playwright не установлен (pip install playwright && python -m playwright install chromium)")
        return []

    import urllib.parse, re
    q_raw = (query or "").strip()
    city_raw = (city or "").strip()
    q_enc    = urllib.parse.quote(q_raw)
    q_param  = urllib.parse.quote_plus(q_raw)
    city_param = urllib.parse.quote_plus(city_raw)

    # если выше в файле уже есть _gr_city_slug — используем его
    city_sub = _gr_city_slug(city_raw) if " _gr_city_slug" in globals() or "_gr_city_slug" in dir() else ""

    def _page_urls(p: int) -> list[str]:
        urls = []
        if city_sub:
            base = f"https://{city_sub}.gorodrabot.ru/{q_enc}"
            urls.append(base if p == 1 else f"{base}?page={p}")
        base2 = f"https://gorodrabot.ru/%D0%B2%D0%B0%D0%BA%D0%B0%D0%BD%D1%81%D0%B8%D0%B8?text={q_param}&city={city_param}"
        urls.append(base2 if p == 1 else f"{base2}&page={p}")
        base3 = f"https://gorodrabot.ru/%D0%B2%D0%B0%D0%BA%D0%B0%D0%BD%D1%81%D0%B8%D0%B8?text={q_param}"
        urls.append(base3 if p == 1 else f"{base3}&page={p}")
        return urls

    def _allowed(u: str) -> bool:
        u = (u or "").strip().lower()
        if not u:
            return False
        # свои страницы gorodrabot -> разрешаем только вакансии/похожие токены
        if "gorodrabot.ru/" in u:
            token_ok = any(t in u for t in ("vacancy", "вакан", "%d0%b2%d0%b0%d0%ba", "/rabota", "/search", "/vacancies"))
            return bool(token_ok)
        # внешние борды — берём
        if any(d in u for d in ("hh.ru", "superjob.ru", "rabota.ru", "zarplata.ru", "trudvsem.ru")):
            return True
        return False

    out, seen = [], set()

    with sync_playwright() as pw:
        ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36")

        browser = pw.chromium.launch(
            headless=not headful,
            args=["--disable-blink-features=AutomationControlled",
                  "--disable-features=IsolateOrigins,site-per-process"]
        )
        ctx_kwargs = dict(
            locale="ru-RU",
            timezone_id="Europe/Moscow",
            user_agent=ua,
            viewport={"width": 1366, "height": 880},
            device_scale_factor=1.0,
        )
        try:
            if state_path:
                from pathlib import Path as _P
                if _P(state_path).exists():
                    ctx_kwargs["storage_state"] = state_path
        except Exception:
            pass

        ctx = browser.new_context(**ctx_kwargs)
        page = ctx.new_page()

        # собираем урлы из XHR JSON
        net_urls: set[str] = set()
        def _from_response(resp):
            try:
                ct = (resp.headers or {}).get("content-type", "")
                if "json" not in ct.lower():
                    return
                if "gorodrabot.ru" not in (resp.url or ""):
                    return
                txt = resp.text()
                for m in re.finditer(r'https?:\/\/[^"\\\s]+', txt):
                    u = (m.group(0) or "").strip()
                    if _allowed(u):
                        net_urls.add(u)
            except Exception:
                pass
        page.on("response", _from_response)

        total_found = 0
        for p in range(1, max(1, int(pages)) + 1):
            found_p = 0
            for url in _page_urls(p):
                hrefs = []               # <-- всегда инициализируем
                net_urls.clear()         # очистим XHR-урлы для каждой выдачи

                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=45000)

                    # попытка закрыть cookie/баннеры (включая iframes)
                    def _try_click_all():
                        selectors = [
                            'button[aria-label="Принять"]',
                            'button:has-text("Принять")',
                            'button:has-text("Понятно")',
                            'button:has-text("Продолжить")',
                            'button:has-text("Да")',
                            'button:has-text("Согласен")',
                            '[role="button"]:has-text("OK")',
                        ]
                        for sel in selectors:
                            try:
                                if page.locator(sel).count() > 0:
                                    page.locator(sel).first.click(timeout=1200)
                                    page.wait_for_timeout(300)
                            except Exception:
                                pass
                        try:
                            for fr in page.frames:
                                for sel in selectors:
                                    try:
                                        if fr.query_selector(sel):
                                            fr.click(sel, timeout=1200)
                                            page.wait_for_timeout(300)
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                    _try_click_all()
                    try:
                        page.wait_for_load_state("networkidle", timeout=8000)
                    except Exception:
                        pass
                    page.wait_for_timeout(500)

                    # плавная прокрутка
                    last_h = 0
                    for _ in range(24):
                        page.mouse.wheel(0, 1400)
                        page.wait_for_timeout(220)
                        try:
                            h = page.evaluate("document.body.scrollHeight")
                        except Exception:
                            h = 0
                        if h <= last_h:
                            break
                        last_h = h

                    # 1) все <a> с "vacan/вакан" в href
                    hrefs = page.evaluate(
                        """() => Array.from(document.querySelectorAll('a'))
                               .map(a => a.href)
                               .filter(h => !!h && /(vacan|вакан)/i.test(h))"""
                    ) or []

                    # 2) значения из data-url / data-href
                    hrefs += page.evaluate(
                        """() => Array.from(document.querySelectorAll('[data-url],[data-href]'))
                               .map(el => el.getAttribute('data-url') || el.getAttribute('data-href'))
                               .filter(h => !!h && /(vacan|вакан|%D0%B2%D0%B0%D0%BA)/i.test(h))"""
                    ) or []

                    # 3) JSON в <script> на странице
                    try:
                        scripts = page.locator("script").all()
                        for sc in scripts[:80]:
                            try:
                                txt = sc.text_content() or ""
                            except Exception:
                                continue
                            for m in re.finditer(r'https?:\/\/[^"\\\s]+', txt):
                                u = m.group(0)
                                if re.search(r'(vacan|вакан)', u, flags=re.I):
                                    hrefs.append(u)
                    except Exception:
                        pass

                    # заголовки (по желанию)
                    titles = {}
                    try:
                        anchors = page.query_selector_all(
                            'a[data-qa="vacancy-title"], a.vacancy-item__title, article a[href*="vacan"], article a[href*="вакан"], main a[href*="vacan"], main a[href*="вакан"]'
                        )
                        for a in anchors:
                            try:
                                u = a.get_attribute("href") or ""
                                if u and not u.startswith("http"):
                                    u = "https://gorodrabot.ru" + u
                                t = (a.inner_text() or "").strip()
                                if u and t:
                                    titles[u] = t
                            except Exception:
                                continue
                    except Exception:
                        pass

                    # объединяем DOM + XHR и фильтруем
                    links = list(set(hrefs) | set(net_urls))
                    added = 0
                    skipped = 0
                    for u in links:
                        u = (u or "").strip()
                        if not u or u in seen:
                            continue
                        if _allowed(u):
                            seen.add(u)
                            out.append({"url": u, "title": titles.get(u)})
                            added += 1
                        else:
                            skipped += 1

                    print(f"[GR][PW] {page.url} +{added} links  (kept={added} skipped={skipped})")

                    # отладочный дамп HTML (по желанию)
                    try:
                        _gr_save_debug(page.content() or "", f"gr_pw_p{p}")
                    except Exception:
                        pass

                    found_p += added
                except Exception as e:
                    print(f"[GR][PW] fail: {e}")

                page.wait_for_timeout(int(max(0.3, pause) * 1000))

            print(f"[GR] page={p} found={found_p}")
            total_found += found_p

        try:
            if state_path:
                ctx.storage_state(path=state_path)
        except Exception:
            pass

        browser.close()

    return out
# -----------------------------------------------------------------------

# =================== СВОД И ВЫВОД ===================
def to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in TEMPLATE_COLS:
        if c not in df.columns: df[c] = None
    df = df[TEMPLATE_COLS]
    if "Примечание" in df.columns:
        df["Примечание"] = df["Примечание"].fillna("")
    if "Ссылка" in df.columns:
        df = df.drop_duplicates(subset=["Ссылка"], keep="first")
    df = df.drop_duplicates(subset=["Должность","Работодатель"], keep="first")
    return df

def main():
    ap = argparse.ArgumentParser(description="Парсер вакансий: hh.ru + gorodrabot.ru + avito (с fallback на Playwright)")
    ap.add_argument("--query", required=True)
    ap.add_argument("--area", type=int, default=1)                   # hh регион (1=Москва)
    ap.add_argument("--city", default="Москва")                      # gorodrabot/avito город
    ap.add_argument("--role", default="повар", help="ключ из FILTERS")
    ap.add_argument("--pages", type=int, default=3)
    ap.add_argument("--per_page", type=int, default=50)
    ap.add_argument("--pause", type=float, default=0.7)
    ap.add_argument("--search_in", default="name", help="name|description|company_name|everything")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--no_filter", action="store_true", help="не применять фильтр по роли (для диагностики)")
    ap.add_argument("--workers", type=int, default=8, help="число потоков для Avito деталей")
    ap.add_argument("--timeout", type=float, default=8.0, help="таймаут HTTP на карточку, сек")
    ap.add_argument("--avito_headful", action="store_true", help="окно браузера для Avito (решить капчу 1 раз)")
    ap.add_argument("--avito_state", default="avito_state.json", help="файл для cookies/storage state")
    ap.add_argument("--no_gr", action="store_true", help="не собирать Город Работ")
    ap.add_argument("--no_avito", action="store_true", help="не собирать Avito")
    ap.add_argument("--gr_headful", action="store_true", help="Город Работ: открыть окно браузера при сборе")
    ap.add_argument("--gr_state", default="gr_state.json", help="Город Работ: файл cookies/storage")

    a = ap.parse_args()
    global _WORKERS, _TIMEOUT
    _WORKERS = max(1, a.workers)
    _TIMEOUT = max(3.0, a.timeout)


    INC_RE, EXC_RE = _compile_filters(a.role)

    # HH
    hh_items = hh_search(a.query, a.area, a.pages, a.per_page, a.pause, a.search_in)
    rows_hh = map_hh(hh_items)

    gr_rows = []
    if not a.no_gr:
        try:
            gr_list = gr_search(a.query, a.city, a.pages, a.pause,
                                headful=getattr(a, "gr_headful", False),
                                state_path=getattr(a, "gr_state", "gr_state.json"))
            gr_rows = map_gr(gr_list)
        except Exception as e:
            print(f"[GR] пропущен: {e}")
    else:
        print("[GR] пропущен (--no_gr)")

    # Avito (опционально)
    rows_avito = []
    if not a.no_avito:
        try:
            av_items = avito_search(a.query, a.city, a.pages, a.pause)
            rows_avito = map_avito(av_items) if av_items else []
        except Exception as e:
            print(f"[Avito] пропущен: {e}")
    else:
        print("[Avito] пропущен (--no_avito)")

    rows = rows_hh + gr_rows + rows_avito
    print(f"HH: {len(rows_hh)} | GR: {len(gr_rows)} | Avito: {len(rows_avito)} | Total before filter: {len(rows)}")

    # Диагностика по доменам
    tmp = pd.DataFrame(rows)
    if not tmp.empty and "Ссылка" in tmp.columns:
        bydom = tmp["Ссылка"].fillna("").str.extract(r"https?://([^/]+)/", expand=False).value_counts().head(5).to_dict()
        print("By domain:", bydom)

    # Фильтр по названию (если не отключен)
    if not a.no_filter:
        rows = [r for r in rows if keep_by_title(str(r.get("Должность", "")), INC_RE, EXC_RE)]
    else:
        print("Фильтр по роли отключен (--no_filter).")

    apply_computed_hourly_rates(rows)
    rate_rows = collect_shift_rate_rows(rows)
    df = to_df(rows)

    # Чистка ссылок перед экспортом (www.www и query)
    if "Ссылка" in df.columns:
        df["Ссылка"] = (
            df["Ссылка"].astype(str)
            .str.replace(r"^https?://(?:www\.){2,}", "https://www.", regex=True)
            .str.replace(r"\?.*$", "", regex=True)
            .str.replace(r"^https?://avito\.ru", "https://www.avito.ru", regex=True)
            .str.replace(r"^https?://m\.avito\.ru", "https://www.avito.ru", regex=True)
        )

    # CSV (для пайплайна)
    df.to_csv(a.out_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(df)} rows -> {a.out_csv}")

    rate_path = Path(str(a.out_csv) + ".rates.json")
    try:
        with open(rate_path, "w", encoding="utf-8") as f:
            json.dump({"items": rate_rows}, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(rate_rows)} rate rows -> {rate_path}")
    except Exception as e:
        print(f"Failed to write rate JSON: {e}")

    # XLSX для просмотра (без \n в заголовках)
    xlsx_path = a.out_csv if str(a.out_csv).lower().endswith(".xlsx") else str(a.out_csv).rsplit(".", 1)[0] + "_view.xlsx"
    df_x = df.rename(columns=lambda c: c.replace("\n", " "))
    df_x.to_excel(xlsx_path, index=False)
    print(f"Wrote {len(df_x)} rows -> {xlsx_path}")

if __name__ == "__main__":
    main()

