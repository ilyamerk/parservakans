import pytest

from fetch_vacancies import WorkingHoursContext, extract_working_hours


def _find_day(by_days, day):
    for item in by_days:
        if item.get("day") == day:
            return item
    return None


def test_working_hours_by_days():
    ctx = WorkingHoursContext(
        text="График работы: Пн–Пт 9:00–18:00, Сб 10–16, Вс выходной",
        url="https://example.com/card",
        selector="#card",
        weight=10,
    )
    result = extract_working_hours([ctx])

    assert result["confidence"] >= 0.9
    assert any("График работы" in raw for raw in result["raw_matches"])
    by_days = result["normalized"]["by_days"]
    assert len(by_days) >= 6

    mon = _find_day(by_days, "Mon")
    sat = _find_day(by_days, "Sat")
    assert mon == {"day": "Mon", "start": "09:00", "end": "18:00"}
    assert sat == {"day": "Sat", "start": "10:00", "end": "16:00"}
    assert _find_day(by_days, "Sun") is None
    assert "Sun" in (result.get("notes") or "")
    assert result["lang"] == "ru"


def test_working_hours_shift_schedule():
    ctx = WorkingHoursContext(
        text="График 2/2 по 12 часов",
        url="https://example.com/shift",
        selector="div.schedule",
        weight=8,
    )
    result = extract_working_hours([ctx])

    assert result["schedule_hint"] == "2/2"
    shift = result["normalized"]["shift_based"]
    assert shift["pattern"] == "2/2"
    assert shift["shift_length_hours"] == 12
    assert result["normalized"]["by_days"] == []
    assert result["confidence"] >= 0.5


def test_working_hours_247():
    ctx = WorkingHoursContext(
        text="Работаем круглосуточно, без выходных",
        url="https://example.com/247",
        selector="section.hours",
        weight=5,
    )
    result = extract_working_hours([ctx])

    assert result["normalized"]["is_247"] is True
    assert result["confidence"] >= 0.9


def test_working_hours_flexible():
    ctx = WorkingHoursContext(
        text="Гибкий график",
        url="https://example.com/flex",
        selector="p.flex",
        weight=3,
    )
    result = extract_working_hours([ctx])

    assert result["raw_matches"] == ["Гибкий график"]
    assert result["normalized"]["by_days"] == []
    assert result["confidence"] <= 0.5


def test_working_hours_english_format():
    ctx = WorkingHoursContext(
        text="Opening hours: Mon–Fri 10am–7pm, Sat 10–4pm",
        url="https://example.com/en",
        selector="#hours",
        weight=9,
    )
    result = extract_working_hours([ctx])

    mon = _find_day(result["normalized"]["by_days"], "Mon")
    sat = _find_day(result["normalized"]["by_days"], "Sat")
    assert mon == {"day": "Mon", "start": "10:00", "end": "19:00"}
    assert sat == {"day": "Sat", "start": "10:00", "end": "16:00"}
    assert result["lang"] == "en"


def test_working_hours_conflict_prefers_card():
    ctx_footer = WorkingHoursContext(
        text="Футер: часы работы 10–18",
        url="https://example.com/job",
        selector="footer",
        weight=2,
    )
    ctx_card = WorkingHoursContext(
        text="Часы работы: Пн–Вс 09:00–20:00",
        url="https://example.com/job",
        selector="#card",
        weight=12,
    )
    result = extract_working_hours([ctx_footer, ctx_card])

    assert any("10–18" in raw for raw in result["raw_matches"])
    assert any("09:00" in raw or "09" in raw for raw in result["raw_matches"])

    by_days = result["normalized"]["by_days"]
    assert _find_day(by_days, "Mon") == {"day": "Mon", "start": "09:00", "end": "20:00"}
    assert result["confidence"] >= 0.9

    selectors = {src["selector"] for src in result["source"]}
    assert {"footer", "#card"}.issubset(selectors)
    assert len(result["log"]) >= 1
