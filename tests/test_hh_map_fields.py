from fetch_vacancies import map_hh


def test_map_hh_extracts_shift_schedule_and_rates_from_text():
    items = [
        {
            "name": "Оператор склада",
            "employer": {"name": "Склад"},
            "published_at": "2026-01-01",
            "salary": {"from": 90000, "to": 100000},
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
    assert row["shift_duration_unresolved"] is False
    assert row["График"] == "2/2"
    assert row["Труд-во"] == "ТК"
    assert row["В час"] == 400.0
    assert row["Средний совокупный доход при графике 2/2 по 12 часов"] == 4800.0
    assert row["hourly_rate_method"].startswith("exact")


def test_map_hh_does_not_compute_hourly_without_shift_duration():
    items = [
        {
            "name": "Кассир",
            "employer": {"name": "Маркет"},
            "published_at": "2026-01-01",
            "salary": {"from": 90000},
            "experience": {"name": "Без опыта"},
            "employment": {"name": "Возможно оформление по ТК РФ или ГПХ"},
            "schedule": {"name": "Гибкий график"},
            "snippet": {
                "requirement": "график 3/3, доход 90000 в месяц",
                "responsibility": "обслуживание",
            },
            "alternate_url": "https://hh.ru/vacancy/101",
        }
    ]

    row = map_hh(items)[0]
    assert row["Длительность смены"] is None
    assert row["shift_duration_source"] == "unresolved"
    assert row["shift_duration_unresolved"] is True
    assert row["Труд-во"] == "ТК / ГПХ"
    assert row["График"] == "3/3"
    assert row["В час"] is None
    assert row["hourly_rate_method"] == "unresolved:monthly_salary_requires_validated_model"
    assert row["Средний совокупный доход при графике 2/2 по 12 часов"] is None


def test_map_hh_extracts_payment_frequency_and_benefits():
    items = [
        {
            "name": "Кладовщик",
            "employer": {"name": "Склад"},
            "published_at": "2026-01-01",
            "salary": {"from": 100000},
            "experience": {"name": "Без опыта"},
            "employment": {"name": "Полная занятость"},
            "schedule": {"name": "Сменный график"},
            "snippet": {
                "requirement": "график 2/2 по 12 часов, выплаты раз в неделю, ДМС и бесплатное питание",
                "responsibility": "работа",
            },
            "alternate_url": "https://hh.ru/vacancy/102",
        }
    ]

    row = map_hh(items)[0]
    assert row["Частота выплат"] == "еженедельно"
    assert row["Льготы"] == "ДМС, питание"
    assert row["В час"] is None
    assert row["hourly_rate_method"] == "unresolved:monthly_salary_requires_validated_model"
