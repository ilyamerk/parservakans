import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import pytest

from avito_source import AvitoCollector, AvitoConfig


TZ = timezone(timedelta(hours=3))


LISTING_PAGE_1 = """
<html lang="ru">
  <body>
    <div data-marker="item">
      <a data-marker="item-title" href="/moskva/vakansii/barista-111111111">Бариста сменный</a>
      <span data-marker="item-price">Оплата за смену 4 800 ₽</span>
      <div data-marker="item-location">Москва</div>
      <div data-marker="item-date">Сегодня 10:00</div>
      <div data-marker="item-badge">Реклама</div>
      <div data-marker="item-snippet">График 2/2 по 12 часов, питание</div>
    </div>
    <div data-marker="item">
      <a data-marker="item-title" href="/moskva/vakansii/operator-222222222">Оператор</a>
      <span data-marker="item-price">Оплата в час 350–450 ₽</span>
      <div data-marker="item-location">Москва</div>
      <div data-marker="item-date">Вчера 12:00</div>
      <div data-marker="item-snippet">Гибкий график, подработка</div>
    </div>
  </body>
</html>
"""


LISTING_PAGE_2 = """
<html lang="ru">
  <body>
    <div data-marker="item">
      <a data-marker="item-title" href="/moskva/vakansii/barista-111111111">Бариста сменный</a>
      <span data-marker="item-price">Оплата за смену 5 200 ₽</span>
      <div data-marker="item-location">Москва</div>
      <div data-marker="item-date">5 часов назад</div>
    </div>
    <div data-marker="item">
      <a data-marker="item-title" href="/moskva/vakansii/barmen-333333333">Бармен</a>
      <div data-marker="item-date">3 дня назад</div>
    </div>
  </body>
</html>
"""


DETAIL_111_V1 = """
<html lang="ru">
  <body>
    <h1 data-marker="item-title">Бариста сменный</h1>
    <div data-marker="seller-info/name">ООО «Кофе»</div>
    <a data-marker="seller-info/link" href="https://www.avito.ru/user/coffee">Профиль</a>
    <ul data-marker="item-params/list">
      <li data-marker="item-params/list-item"><span class="param-label">Тип занятости:</span><span class="param-value">Полная занятость</span></li>
      <li data-marker="item-params/list-item"><span class="param-label">График работы:</span><span class="param-value">2/2 по 12 часов</span></li>
      <li data-marker="item-params/list-item"><span class="param-label">Опыт работы:</span><span class="param-value">Не требуется</span></li>
      <li data-marker="item-params/list-item"><span class="param-label">Адрес:</span><span class="param-value">Москва, ул. Пушкина, 1</span></li>
    </ul>
    <div data-marker="item-description/text">
      <p>Оплата за смену 4800 ₽, график 2/2 по 12 часов.</p>
      <h2>Обязанности</h2>
      <p>Приготовление напитков</p>
      <h2>Условия</h2>
      <ul><li>Питание</li><li>Форма</li></ul>
    </div>
    <time itemprop="datePublished" datetime="2024-06-01T10:00:00+03:00"></time>
    <span data-marker="item-view/total">Просмотров: 120</span>
    <span data-marker="item-view/favorites">В избранном: 4</span>
    <div data-marker="seller-info/contact">Контакт: Анна</div>
    <ol itemprop="breadcrumb">
      <li itemprop="itemListElement"><span itemprop="name">Работа</span></li>
      <li itemprop="itemListElement"><span itemprop="name">Ресторанный бизнес</span></li>
    </ol>
    <div data-marker="gallery/image"><img src="1.jpg"/></div>
    <div data-marker="gallery/image"><img src="2.jpg"/></div>
  </body>
</html>
"""


DETAIL_111_V2 = DETAIL_111_V1.replace("4800", "5200").replace("Просмотров: 120", "Просмотров: 240")


DETAIL_222 = """
<html lang="ru">
  <body>
    <h1 data-marker="item-title">Оператор</h1>
    <div data-marker="seller-info/name">ИП Иванов</div>
    <div data-marker="item-description/text">
      <p>Оплата в час 350–450 ₽. Гибкий график, подработка.</p>
    </div>
    <ul data-marker="item-params/list">
      <li data-marker="item-params/list-item"><span class="param-label">Тип занятости:</span><span class="param-value">Частичная занятость</span></li>
      <li data-marker="item-params/list-item"><span class="param-label">Опыт работы:</span><span class="param-value">Опыт работы от 1 года</span></li>
    </ul>
    <time itemprop="datePublished">Вчера 12:00</time>
    <span data-marker="item-view/total">Просмотров: 45</span>
  </body>
</html>
"""


class FakeResponse:
    def __init__(self, url: str, text: str, status: int = 200) -> None:
        self.url = url
        self._text = text
        self.status_code = status
        self.headers = {}

    @property
    def text(self) -> str:
        return self._text


class FakeFetcher:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = defaultdict(int)

    def __call__(self, url: str) -> FakeResponse:
        entry = self.mapping.get(url)
        if entry is None:
            raise AssertionError(f"Unexpected URL {url}")
        call_index = self.calls[url]
        self.calls[url] += 1
        if callable(entry):
            entry = entry(call_index)
        status = entry.get("status", 200)
        text = entry.get("text", "")
        return FakeResponse(url, text, status=status)


@pytest.fixture()
def fake_fetcher():
    listing_url_1 = "https://www.avito.ru/moskva/vakansii?p=1&q=%D0%B1%D0%B0%D1%80%D0%B8%D1%81%D1%82%D0%B0"
    listing_url_2 = "https://www.avito.ru/moskva/vakansii?p=2&q=%D0%B1%D0%B0%D1%80%D0%B8%D1%81%D1%82%D0%B0"
    detail_111 = "https://www.avito.ru/moskva/vakansii/barista-111111111"
    detail_222 = "https://www.avito.ru/moskva/vakansii/operator-222222222"
    detail_333 = "https://www.avito.ru/moskva/vakansii/barmen-333333333"

    mapping = {
        listing_url_1: {"text": LISTING_PAGE_1},
        listing_url_2: {"text": LISTING_PAGE_2},
        detail_111: lambda call_index: {"text": DETAIL_111_V1 if call_index == 0 else DETAIL_111_V2},
        detail_222: {"text": DETAIL_222},
        detail_333: {"status": 404, "text": "Удалено"},
    }
    return FakeFetcher(mapping)


def _records_by_id(records):
    return {rec["external_id"]: rec for rec in records}


def test_collect_avito_vacancies(fake_fetcher, caplog, tmp_path):
    config = AvitoConfig(
        regions=["Москва"],
        queries=["бариста"],
        categories=[],
        max_pages_per_feed=2,
        qa_output_dir=tmp_path,
    )
    now = datetime(2024, 6, 2, 15, 0, tzinfo=TZ)
    collector = AvitoCollector(config=config, fetcher=fake_fetcher, now=now)

    with caplog.at_level("INFO"):
        records = collector.collect()

    assert "Avito stats" in " ".join(caplog.messages)
    by_id = _records_by_id(records)
    assert set(by_id) == {111111111, 222222222, 333333333}

    barista = by_id[111111111]
    assert barista["is_active"] is True
    assert barista["is_promoted"] is True
    assert barista["salary_period"] == "per_shift"
    assert barista["salary_from"] == pytest.approx(5200)
    assert barista["views_count"] == 240
    assert barista["schedule_hint"] == "2/2"
    assert barista["employment_type"].lower().startswith("полная")
    assert barista["benefits"] == ["Питание", "Форма"]
    assert barista["contact"]["raw"].startswith("Контакт")
    assert barista["photos_count"] == 2
    assert barista["category_path"] == ["Работа", "Ресторанный бизнес"]
    assert barista["posted_at"].isoformat() == "2024-06-01T10:00:00+03:00"
    assert barista["working_hours"]["normalized"]["shift_based"]["pattern"] == "2/2"
    raw_matches = "\n".join(barista["working_hours"]["raw_matches"])
    assert "2/2 по 12 часов" in raw_matches

    operator = by_id[222222222]
    assert operator["salary_period"] == "per_hour"
    assert operator["salary_from"] == pytest.approx(350)
    assert operator["salary_to"] == pytest.approx(450)
    assert operator["salary_currency"] == "RUB"
    assert any("Гибкий график" in raw for raw in operator["working_hours"]["raw_matches"])
    assert operator["working_hours"]["confidence"] <= 0.5
    assert operator["posted_at"].date() == datetime(2024, 6, 1, tzinfo=TZ).date()

    removed = by_id[333333333]
    assert removed["is_active"] is False
    assert removed["last_seen_at"] == now
    assert removed["diagnostics"]["detail_status"] == 404

    # deduplication: second fetch updates values
    assert barista["diagnostics"]["deduplicated"] == 1

    # QA samples limited and persisted
    assert 0 < len(collector.qa_samples) <= 3
    for sample in collector.qa_samples:
        assert sample["listing_html"]
        assert "record" in sample

    dumped = list(tmp_path.glob("*_parsed.json"))
    assert dumped, "QA samples were not written"
    parsed_example = json.loads(dumped[0].read_text(encoding="utf-8"))
    assert parsed_example["source"] == "avito"

