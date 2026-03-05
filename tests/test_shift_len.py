import pytest

from fetch_vacancies import extract_shift_len, ShiftLength


@pytest.mark.parametrize(
    "text, expected_hours",
    [
        ("Смены по 11 часов, график 2/2", 11.0),
        ("Работа с 9:00 до 21:00, оплата вовремя", 12.0),
        ("Смены 10-12 часов", 11.0),
        ("12-часовая смена", 12.0),
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
    assert extract_shift_len(text) is None


def test_legacy_row_does_not_default_shift_length_to_12_hours():
    from fetch_vacancies import _legacy_row_from_avito_record

    row = _legacy_row_from_avito_record(
        {
            "title": "Оператор",
            "salary_period": "per_shift",
            "salary_from": 3000,
            "salary_to": 3000,
            "schedule_hint": "2/2",
            "working_hours": {"normalized": {"shift_based": {"pattern": "2/2"}}},
        }
    )

    assert row["Длительность \nсмены"] is None


def test_legacy_row_extracts_shift_length_from_schedule_hint_text():
    from fetch_vacancies import _legacy_row_from_avito_record

    row = _legacy_row_from_avito_record(
        {
            "title": "Оператор",
            "salary_period": "per_shift",
            "salary_from": 3000,
            "salary_to": 3000,
            "schedule_hint": "2/2 по 10 часов",
            "working_hours": {"normalized": {"shift_based": {"pattern": "2/2"}}},
        }
    )

    assert row["Длительность \nсмены"] == pytest.approx(10.0)
