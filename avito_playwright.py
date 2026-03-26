"""Avito vacancy collector using Playwright (Chromium, headless).

This module provides AvitoPlaywrightCollector that renders Avito pages
with a real browser to bypass JavaScript rendering. Returns normalized
records compatible with _legacy_row_from_avito_record() in fetch_vacancies.py.
"""

from __future__ import annotations

import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote


# --- Region slug mapping ---
_REGION_MAP = {
    "москва": "moskva",
    "санкт-петербург": "sankt-peterburg",
    "спб": "sankt-peterburg",
    "екатеринбург": "ekaterinburg",
    "новосибирск": "novosibirsk",
    "казань": "kazan",
    "нижний новгород": "nizhniy_novgorod",
    "челябинск": "chelyabinsk",
    "самара": "samara",
    "ростов-на-дону": "rostov-na-donu",
    "уфа": "ufa",
    "красноярск": "krasnoyarsk",
    "пермь": "perm",
    "воронеж": "voronezh",
    "волгоград": "volgograd",
}


def _to_region_slug(region: str) -> str:
    key = region.strip().lower()
    if key in _REGION_MAP:
        return _REGION_MAP[key]
    # Already a slug (latin)
    if re.match(r"^[a-z0-9_-]+$", key):
        return key
    # Fallback: simple transliteration
    _tr = str.maketrans({
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e",
        "ё": "e", "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k",
        "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
        "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
        "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "",
        "э": "e", "ю": "yu", "я": "ya",
    })
    slug = key.translate(_tr)
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug


def _parse_salary(text: str) -> dict:
    """Extract salary_from, salary_to, salary_currency, salary_period from text."""
    result: Dict[str, object] = {
        "salary_from": None,
        "salary_to": None,
        "salary_currency": "RUB",
        "salary_period": "per_month",
    }
    if not text:
        return result

    clean = text.replace("\xa0", " ").replace(" ", " ").strip()

    # Detect period
    low = clean.lower()
    if any(m in low for m in ["/ч", "в час", "час."]):
        result["salary_period"] = "per_hour"
    elif any(m in low for m in ["за смену", "/смен", "смена"]):
        result["salary_period"] = "per_shift"
    elif any(m in low for m in ["в день", "/день"]):
        result["salary_period"] = "per_day"

    # Extract numbers
    numbers = re.findall(r"[\d]+(?:\s[\d]+)*", clean)
    parsed = []
    for n in numbers:
        try:
            val = int(n.replace(" ", ""))
            if val > 0:
                parsed.append(val)
        except ValueError:
            pass

    if len(parsed) == 1:
        result["salary_from"] = parsed[0]
    elif len(parsed) >= 2:
        result["salary_from"] = min(parsed[0], parsed[1])
        result["salary_to"] = max(parsed[0], parsed[1])

    return result


class AvitoPlaywrightCollector:
    """Collect Avito vacancies using Playwright with Chromium in headless mode."""

    # Primary item selectors
    _ITEM_SELECTORS = [
        '[data-marker="item"]',
        'div[class*="iva-item-root"]',
        'div[data-marker="catalog-serp"] > div',
    ]

    # Title/link selectors
    _TITLE_SELECTORS = [
        '[data-marker="item-title"]',
        'a[itemprop="url"]',
        'h3 a',
        'a[href*="/vakansii/"]',
    ]

    @staticmethod
    def _find_chromium() -> Optional[str]:
        """Search for installed Chromium in ms-playwright cache."""
        import os
        cache = Path(os.path.expanduser("~/.cache/ms-playwright"))
        if not cache.exists():
            return None
        for d in sorted(cache.iterdir(), reverse=True):
            if d.name.startswith("chromium-"):
                chrome = d / "chrome-linux" / "chrome"
                if chrome.exists():
                    return str(chrome)
        return None

    # Price selectors
    _PRICE_SELECTORS = [
        '[data-marker="item-price"]',
        'meta[itemprop="price"]',
        'span[class*="price"]',
    ]

    def collect(
        self,
        query: str,
        region: str = "Москва",
        pages: int = 3,
        per_page: int = 50,
    ) -> List[Dict[str, object]]:
        """Collect vacancies from Avito using Playwright.

        Returns list of dicts compatible with AvitoCollector record format
        (suitable for _legacy_row_from_avito_record).
        """
        from playwright.sync_api import sync_playwright

        region_slug = _to_region_slug(region)
        results: List[Dict[str, object]] = []
        debug_dir = Path("Exports/_debug")
        debug_dir.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as pw:
            # Try default launch first; fall back to known Chromium paths
            launch_kwargs = {"headless": True}
            try:
                browser = pw.chromium.launch(**launch_kwargs)
            except Exception as launch_err:
                print(f"[Avito/pw] default launch failed: {launch_err}")
                # Search for installed Chromium in common paths
                chromium_path = self._find_chromium()
                if chromium_path:
                    print(f"[Avito/pw] found Chromium at {chromium_path}")
                    launch_kwargs["executable_path"] = chromium_path
                    browser = pw.chromium.launch(**launch_kwargs)
                else:
                    # Auto-install Chromium and retry
                    import subprocess as _sp
                    print("[Avito/pw] Chromium not found, installing via playwright install chromium ...")
                    _sp.run(
                        ["python", "-m", "playwright", "install", "chromium"],
                        check=True,
                    )
                    # After install, try find again then default launch
                    chromium_path = self._find_chromium()
                    if chromium_path:
                        launch_kwargs["executable_path"] = chromium_path
                    browser = pw.chromium.launch(**launch_kwargs)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1920, "height": 1080},
                locale="ru-RU",
            )
            page = context.new_page()

            for page_num in range(1, pages + 1):
                url = (
                    f"https://www.avito.ru/{region_slug}/vakansii"
                    f"?q={quote(query)}&p={page_num}"
                )
                print(f"[Avito/pw] GET {url}")

                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                except Exception as e:
                    print(f"[Avito/pw] page.goto failed: {e}")
                    continue

                # Wait for vacancy cards to appear
                cards_found = False
                for selector in self._ITEM_SELECTORS:
                    try:
                        page.wait_for_selector(selector, timeout=15000)
                        cards_found = True
                        break
                    except Exception:
                        continue

                if not cards_found:
                    # Save debug screenshot
                    screenshot_path = debug_dir / f"avito_page_{page_num}.png"
                    try:
                        page.screenshot(path=str(screenshot_path))
                        print(f"[Avito/pw] debug screenshot saved: {screenshot_path}")
                    except Exception:
                        pass
                    print(f"[Avito/pw] no cards found on page {page_num}, skipping")
                    if page_num == 1:
                        break
                    continue

                # Scroll down to trigger lazy loading
                page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                page.wait_for_timeout(1000)
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(1000)

                # Extract cards
                page_cards = self._extract_cards(page)
                results.extend(page_cards)

                if len(results) >= per_page * pages:
                    break

                # Random pause between pages
                if page_num < pages:
                    pause = random.uniform(2.0, 3.0)
                    time.sleep(pause)

            browser.close()

        return results

    def _extract_cards(self, page) -> List[Dict[str, object]]:
        """Extract vacancy cards from the current page."""
        cards: List[Dict[str, object]] = []

        # Find all item containers
        items = []
        for selector in self._ITEM_SELECTORS:
            items = page.query_selector_all(selector)
            if items:
                break

        for item in items:
            try:
                record = self._parse_card(item)
                if record and record.get("title"):
                    cards.append(record)
            except Exception as exc:
                print(f"[Avito/pw] _parse_card error: {exc}")
                continue

        return cards

    def _parse_card(self, item) -> Optional[Dict[str, object]]:
        """Parse a single card element into a normalized record dict."""
        title = None
        url = None
        company = None
        salary_text = None

        # Extract title and URL
        for sel in self._TITLE_SELECTORS:
            el = item.query_selector(sel)
            if el:
                title = (el.inner_text() or "").strip()
                href = el.get_attribute("href") or ""
                if href:
                    if href.startswith("/"):
                        url = f"https://www.avito.ru{href}"
                    elif href.startswith("http"):
                        url = href
                if title:
                    break

        # Fallback: try h3 tag for title
        if not title:
            h3 = item.query_selector("h3")
            if h3:
                title = (h3.inner_text() or "").strip()
                link = h3.query_selector("a")
                if link and not url:
                    href = link.get_attribute("href") or ""
                    if href.startswith("/"):
                        url = f"https://www.avito.ru{href}"
                    elif href.startswith("http"):
                        url = href

        if not title:
            return None

        # Filter: only vacancy-like URLs
        if url and "/vakansii/" not in url:
            return None

        # Extract salary/price
        for sel in self._PRICE_SELECTORS:
            el = item.query_selector(sel)
            if el:
                if sel.startswith("meta"):
                    salary_text = el.get_attribute("content") or ""
                else:
                    salary_text = (el.inner_text() or "").strip()
                if salary_text:
                    break

        # Extract company name (various possible selectors)
        for sel in [
            '[data-marker="item-line"]',
            'span[class*="company"]',
            'p[class*="company"]',
            'div[class*="iva-item-content"] > p',
        ]:
            el = item.query_selector(sel)
            if el:
                text = (el.inner_text() or "").strip()
                if text and len(text) < 200:
                    company = text
                    break

        # Extract external_id from URL
        external_id = None
        if url:
            m = re.search(r"-(\d{6,})$", url.split("?")[0])
            if m:
                external_id = int(m.group(1))

        # Parse salary numbers
        salary_info = _parse_salary(salary_text or "")

        return {
            "source": "avito",
            "external_id": external_id,
            "url_listing": url,
            "url_detail": url,
            "title": title,
            "employer_name": company,
            "salary_from": salary_info.get("salary_from"),
            "salary_to": salary_info.get("salary_to"),
            "salary_currency": salary_info.get("salary_currency", "RUB"),
            "salary_period": salary_info.get("salary_period", "per_month"),
            "salary_text": salary_text,
            "posted_at": None,
            "posted_at_raw": None,
            "schedule_hint": None,
            "experience_required": {},
            "employment_type": None,
            "benefits": [],
            "duties_raw": None,
            "working_hours": {},
            "is_active": True,
            "is_promoted": False,
            "is_featured": False,
            "diagnostics": {"item_selector": ["playwright"]},
        }
