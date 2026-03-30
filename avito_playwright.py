"""Avito vacancy collector via internal JSON API (no browser needed).

This module provides AvitoPlaywrightCollector that fetches Avito vacancies
using their internal JSON API with requests. Returns normalized records
compatible with _legacy_row_from_avito_record() in fetch_vacancies.py.

Class name kept as AvitoPlaywrightCollector for backward compatibility.
"""

import requests
import re
import time
import random
import json
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import quote

DEBUG_DIR = Path("Exports/_debug")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
]

# locationId для городов
LOCATION_IDS = {
    "москва": 637640,
    "санкт-петербург": 653240,
    "екатеринбург": 653060,
    "новосибирск": 653100,
    "казань": 653070,
    "нижний новгород": 653090,
    "самара": 653120,
    "уфа": 653150,
    "красноярск": 653080,
    "пермь": 653105,
    "воронеж": 653050,
    "волгоград": 653040,
}

REGION_SLUGS = {
    "москва": "moskva",
    "санкт-петербург": "sankt-peterburg",
    "екатеринбург": "ekaterinburg",
    "новосибирск": "novosibirsk",
    "казань": "kazan",
    "нижний новгород": "nizhniy_novgorod",
    "самара": "samara",
    "уфа": "ufa",
    "красноярск": "krasnoyarsk",
    "пермь": "perm",
    "воронеж": "voronezh",
    "волгоград": "volgograd",
}


class AvitoPlaywrightCollector:
    """Сбор вакансий Avito через внутреннее JSON API (без браузера)."""

    AVITO_CATEGORY_VAKANSII = 110

    def _make_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.avito.ru/",
            "x-requested-with": "XMLHttpRequest",
            "sec-ch-ua": '"Chromium";v="122", "Not(A:Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "Connection": "keep-alive",
        })
        return session

    def _parse_salary(self, price_str: str) -> tuple:
        """Извлечь salary_from и salary_to из строки типа '80 000–120 000 ₽'"""
        if not price_str:
            return None, None
        nums = re.findall(r'[\d\s]+', price_str.replace('\u202f', '').replace('\xa0', ''))
        vals = []
        for n in nums:
            n = n.strip().replace(' ', '')
            if n.isdigit() and int(n) > 0:
                vals.append(int(n))
        if not vals:
            return None, None
        if len(vals) == 1:
            return float(vals[0]), None
        return float(min(vals)), float(max(vals))

    def _parse_items_from_json(self, data: dict) -> List[Dict]:
        """Извлечь вакансии из JSON-ответа API."""
        records = []
        # Avito API возвращает items в разных полях в зависимости от версии
        items = (
            data.get("items")
            or data.get("result", {}).get("items")
            or data.get("data", {}).get("items")
            or []
        )
        print(f"[Avito API] найдено items в JSON: {len(items)}")

        for item in items:
            if not isinstance(item, dict):
                continue
            ext_id = item.get("id") or item.get("itemId")
            title = item.get("title") or item.get("name") or ""
            url_path = item.get("urlPath") or item.get("url") or ""
            if url_path and not url_path.startswith("http"):
                url = f"https://www.avito.ru{url_path}"
            else:
                url = url_path or f"https://www.avito.ru/items/{ext_id}"

            # Зарплата
            price_info = item.get("priceDetailed") or item.get("price") or {}
            if isinstance(price_info, dict):
                price_str = price_info.get("string") or str(price_info.get("value", ""))
            else:
                price_str = str(price_info) if price_info else ""
            sal_from, sal_to = self._parse_salary(price_str)

            # Компания
            seller = item.get("seller") or item.get("company") or {}
            company = ""
            if isinstance(seller, dict):
                company = seller.get("name") or seller.get("title") or ""

            records.append({
                "source": "avito",
                "external_id": ext_id,
                "title": title,
                "employer_name": company,
                "salary_from": sal_from,
                "salary_to": sal_to,
                "salary_currency": "RUB",
                "salary_period": "per_month",
                "url_detail": url,
                "url_listing": url,
                "posted_at": None,
                "posted_at_raw": item.get("sortTimeStamp") or item.get("time"),
                "schedule_hint": None,
                "experience_required": {},
                "employment_type": None,
                "benefits": [],
                "working_hours": {},
                "duties_raw": item.get("description") or item.get("snippet"),
                "is_active": True,
                "diagnostics": {"item_selector": ["api"]},
            })
        return records

    def collect(
        self,
        query: str,
        region: str = "Москва",
        pages: int = 3,
        per_page: int = 25,
    ) -> List[Dict]:
        session = self._make_session()
        city_key = region.strip().lower()
        location_id = LOCATION_IDS.get(city_key, 637640)
        region_slug = REGION_SLUGS.get(city_key, "moskva")
        query_encoded = quote(query)
        results = []
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

        # Сначала сделай обычный GET на страницу поиска чтобы получить куки
        warmup_url = f"https://www.avito.ru/{region_slug}/vakansii?q={query_encoded}"
        try:
            print(f"[Avito] warmup GET: {warmup_url}")
            warmup = session.get(warmup_url, timeout=15)
            print(f"[Avito] warmup status: {warmup.status_code}")
            # Сохрани для диагностики
            (DEBUG_DIR / "avito_warmup.html").write_text(warmup.text[:50000], encoding="utf-8")
        except Exception as e:
            print(f"[Avito] warmup failed: {e}")

        time.sleep(random.uniform(2.0, 3.0))

        # Варианты API endpoint — попробуй все, используй первый рабочий
        api_templates = [
            f"https://www.avito.ru/api/11/items?categoryId={self.AVITO_CATEGORY_VAKANSII}&locationId={location_id}&query={query_encoded}&page={{page}}&limit={per_page}",
            f"https://www.avito.ru/api/9/items?categoryId={self.AVITO_CATEGORY_VAKANSII}&locationId={location_id}&query={query_encoded}&page={{page}}&limit={per_page}",
            f"https://www.avito.ru/{region_slug}/vakansii?q={query_encoded}&p={{page}}&forceLocation=1",
        ]

        working_template = None

        for page in range(1, pages + 1):
            got_data = False

            # Если уже знаем рабочий шаблон — используем его
            if working_template:
                templates_to_try = [working_template.format(page=page)]
            else:
                templates_to_try = [t.format(page=page) for t in api_templates]

            for url in templates_to_try:
                try:
                    print(f"[Avito] GET page={page}: {url[:100]}")
                    r = session.get(url, timeout=20)
                    print(f"[Avito] status={r.status_code}, content-type={r.headers.get('content-type', '?')[:50]}")

                    # Сохрани первую страницу для диагностики
                    if page == 1:
                        suffix = "json" if "json" in r.headers.get("content-type", "") else "html"
                        (DEBUG_DIR / f"avito_page1.{suffix}").write_bytes(r.content[:100000])

                    if r.status_code == 429:
                        print("[Avito] 429 — жду 15 сек")
                        time.sleep(15)
                        continue

                    if r.status_code == 200:
                        content_type = r.headers.get("content-type", "")
                        if "json" in content_type:
                            data = r.json()
                            page_records = self._parse_items_from_json(data)
                            if page_records:
                                results.extend(page_records)
                                print(f"[Avito] page={page} +{len(page_records)} вакансий (итого {len(results)})")
                                if not working_template:
                                    # Сохраняем шаблон с placeholder для page
                                    for t in api_templates:
                                        if url == t.format(page=page):
                                            working_template = t
                                            break
                                got_data = True
                                break
                            else:
                                print(f"[Avito] JSON пришёл но items пустые, пробую следующий endpoint")
                        else:
                            print(f"[Avito] не JSON ответ, content-type={content_type}")

                except requests.RequestException as e:
                    print(f"[Avito] request error: {e}")

            if not got_data and page == 1:
                print("[Avito] Страница 1 не дала данных — проверь Exports/_debug/avito_page1.*")

            if page < pages:
                pause = random.uniform(2.0, 4.0)
                print(f"[Avito] пауза {pause:.1f}с")
                time.sleep(pause)

        print(f"[Avito] итого собрано: {len(results)} вакансий")
        return results
