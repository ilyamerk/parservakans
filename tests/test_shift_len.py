import pytest

from fetch_vacancies import (
    ShiftLength,
    compute_hourly_rate,
    compute_shift_income_total,
    extract_employment_type,
    extract_schedule,
    extract_shift_len,
)


@pytest.mark.parametrize(
    "text, expected_hours",
    [
        ("Смены по 11 часов, график 2/2", 11.0),
        ("Работа с 9:00 до 21:00, оплата вовремя", 12.0),
        ("12-часовая смена", 12.0),
        ("2/2 по 12 часов", 12.0),
        ("смены по 10,5 часов", 10.5),
    ],
)
def test_extract_shift_len_numeric(text, expected_hours):
    result = extract_shift_len(text)
    assert isinstance(result, ShiftLength)
    assert result.hours == pytest.approx(expected_hours)


def test_extract_shift_len_prefers_shift_over_break():
    text = "Перерыв 1 час, смена 8 часов"
    result = extract_shift_len(text)
    assert result is not None
    assert result.hours == pytest.approx(8.0)


def test_extract_shift_len_ignores_experience_range():
    text = "Опыт работы от 1 до 3 лет"
    result = extract_shift_len(text)
    assert result is not None
    assert result.hours is None
    assert result.source == "unresolved"


def test_shift_rate_and_explicit_shift_len():
    text = "2/2 по 12 часов, оплата 4800 за смену"
    sl = extract_shift_len(text)
    hour, method, _ = compute_hourly_rate(None, 4800, sl)
    assert hour == pytest.approx(400.0)
    assert method.startswith("exact")


def test_time_range_shift_len_day_and_night():
    day = extract_shift_len("смена с 08:00 до 20:00")
    night = extract_shift_len("ночная смена с 20:00 до 08:00")
    assert day.hours == pytest.approx(12.0)
    assert night.hours == pytest.approx(12.0)


def test_decimal_shift_len_parsing():
    sl = extract_shift_len("смены по 10,5 часов, оплата 4200")
    hour, method, _ = compute_hourly_rate(None, 4200, sl)
    assert sl.hours == pytest.approx(10.5)
    assert hour == pytest.approx(400.0)
    assert method.startswith("exact")


def test_unresolved_shift_len_keeps_hourly_unresolved():
    sl = extract_shift_len("график 3/3, доход 90000 в месяц")
    hour, method, _ = compute_hourly_rate(None, None, sl)
    assert sl.hours is None
    assert hour is None
    assert method.startswith("unresolved")


def test_multiple_shift_variants_marked_ambiguous():
    sl = extract_shift_len("дневные смены по 12 часов, ночные смены по 10 часов")
    assert sl.ambiguous is True
    hour, method, notes = compute_hourly_rate(None, 5500, sl)
    assert hour is None
    assert method == "unresolved:ambiguous_shift_duration"
    assert notes


def test_shift_rate_from_time_range_5500():
    sl = extract_shift_len("смена с 08:00 до 20:00, 5500 за смену")
    hour, _, _ = compute_hourly_rate(None, 5500, sl)
    assert hour == pytest.approx(458.33)


def test_shift_rate_from_night_range_6000():
    sl = extract_shift_len("ночная смена с 20:00 до 08:00, 6000 за смену")
    hour, _, _ = compute_hourly_rate(None, 6000, sl)
    assert hour == pytest.approx(500.0)


def test_shift_income_total():
    sl = ShiftLength(hours=12.0)
    total, method = compute_shift_income_total(400.0, sl)
    assert total == pytest.approx(4800.0)
    assert method.startswith("exact")


def test_employment_type_tk_gph_and_both():
    assert extract_employment_type("официальное трудоустройство по ТК РФ")[0] == "ТК"
    assert extract_employment_type("оформление по ГПХ")[0] == "ГПХ"
    assert extract_employment_type("возможно оформление по ТК РФ или ГПХ")[0] == "ТК|ГПХ"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("график 5/2", "5/2"),
        ("работа по графику 2/2", "2/2"),
        ("вахта 15/15", "15/15"),
        ("сменный график", "сменный"),
        ("гибкий график", "гибкий"),
    ],
)
def test_extract_schedule(text, expected):
    value, _ = extract_schedule(text)
    assert value == expected
