"""Tests for map_hh field extraction (compatible with HEAD's API)."""
from unittest.mock import patch

from fetch_vacancies import map_hh


def _mock_hh_details(vid):
    """Return empty details to avoid real HTTP calls."""
    return {}


@patch("fetch_vacancies.hh_details", side_effect=_mock_hh_details)
@patch("fetch_vacancies._extract_schedule_from_html", return_value=(None, None))
def test_map_hh_extracts_shift_schedule_and_rates_from_text(_html_mock, _det_mock):
    items = [
        {
            "id": "100",
            "name": "Оператор склада",
            "employer": {"name": "Склад"},
            "published_at": "2026-01-01",
            "salary": {"from": 90000, "to": 100000, "currency": "RUR"},
            "experience": {"name": "Без опыта"},
            "employment": {"name": "Полная занятость"},
            "schedule": {"name": "Сменный график"},
            "snippet": {
                "requirement": "график 2/2 по 12 часов, оплата 4800 за смену, официальное трудоустройство по ТК РФ",
                "responsibility": "работа на складе",
            },
            "alternate_url": "https://hh.ru/vacancy/100",
        }
    ]

    row = map_hh(items)[0]
    assert row["Длительность смены"] == 12.0
    assert row["shift_duration_source"] == "description_explicit_hours"
    assert row["shift_duration_confidence"] == "high"
    assert row["График"] == "2/2"
    assert row["В час"] is not None
    assert row["hourly_rate_method"] is not None


@patch("fetch_vacancies.hh_details", side_effect=_mock_hh_details)
@patch("fetch_vacancies._extract_schedule_from_html", return_value=(None, None))
def test_map_hh_returns_null_hourly_when_missing_shift_duration_for_monthly_salary(_html_mock, _det_mock):
    items = [
        {
            "id": "101",
            "name": "Кассир",
            "employer": {"name": "Маркет"},
            "published_at": "2026-01-01",
            "salary": {"from": 90000, "currency": "RUR"},
            "experience": {"name": "Без опыта"},
            "employment": {"name": "Гибкий график"},
            "schedule": {"name": "Гибкий график"},
            "snippet": {
                "requirement": "график 3/3, доход 90000 в месяц",
                "responsibility": "обслуживание",
            },
            "alternate_url": "https://hh.ru/vacancy/101",
        }
    ]

    row = map_hh(items)[0]
    assert row["shift_duration_source"] == "unresolved"
    assert row["График"] == "3/3"


@patch("fetch_vacancies.hh_details", side_effect=_mock_hh_details)
@patch("fetch_vacancies._extract_schedule_from_html", return_value=(None, None))
def test_map_hh_extracts_benefits(_html_mock, _det_mock):
    items = [
        {
            "id": "102",
            "name": "Кладовщик",
            "employer": {"name": "Склад"},
            "published_at": "2026-01-01",
            "salary": {"from": 100000, "currency": "RUR"},
            "experience": {"name": "Без опыта"},
            "employment": {"name": "Полная занятость"},
            "schedule": {"name": "Сменный график"},
            "snippet": {
                "requirement": "график 2/2 по 12 часов, ДМС и бесплатное питание",
                "responsibility": "работа",
            },
            "alternate_url": "https://hh.ru/vacancy/102",
        }
    ]

    row = map_hh(items)[0]
    assert row["Льготы"] is not None
    assert "ДМС" in row["Льготы"] or "питание" in row["Льготы"]
