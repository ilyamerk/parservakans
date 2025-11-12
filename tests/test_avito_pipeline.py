from datetime import date, datetime, timezone, timedelta

import pandas as pd

from avito_pipeline import (
    AvitoPipelineConfig,
    build_analytics,
    normalize_records,
)


TZ = timezone(timedelta(hours=3))


SAMPLE_RECORD = {
    "external_id": 111111111,
    "url_detail": "https://www.avito.ru/moskva/vakansii/barista-111111111",
    "url_listing": "https://www.avito.ru/moskva/vakansii?p=1&q=%D0%B1%D0%B0%D1%80%D0%B8%D1%81%D1%82%D0%B0",
    "title": "Бариста сменный",
    "location_city": "Москва",
    "location_area": None,
    "employer_name": "ООО «Кофе»",
    "salary_from": 4800,
    "salary_to": 4800,
    "salary_currency": "RUB",
    "salary_period": "per_shift",
    "schedule_hint": "2/2",
    "employment_type": "Полная занятость",
    "experience_required": {"raw": "Не требуется", "normalized": None},
    "duties_raw": "Приготовление напитков",
    "conditions_raw": "Питание\nФорма",
    "benefits": ["Питание", "Форма"],
    "requirements_raw": None,
    "contact": {"raw": "Контакт: Анна"},
    "working_hours": {
        "raw_matches": ["Оплата за смену 4800 ₽, график 2/2 по 12 часов."],
        "normalized": {
            "shift_based": {
                "pattern": "2/2",
                "shift_length_hours": 12.0,
            }
        },
        "schedule_hint": "2/2",
    },
    "posted_at": datetime(2024, 6, 1, 10, 0, tzinfo=TZ),
    "diagnostics": {"detail_status": 200},
    "is_active": True,
}


def make_config() -> AvitoPipelineConfig:
    return AvitoPipelineConfig(
        queries=["бариста"],
        city="Москва",
        region="Москва",
        pages_limit=2,
        schedule_defaults={
            "2/2": {"shifts_per_month": 15, "hours_per_shift": 12},
            "5/2": {"shifts_per_month": 23, "hours_per_shift": 8},
        },
        delay_range=(0.0, 0.0),
    )


def test_normalize_records_basic():
    config = make_config()
    df, logs = normalize_records([SAMPLE_RECORD], config, collected_at=date(2024, 6, 3))

    assert list(df.columns)[:5] == [
        "Источник",
        "Площадка ID",
        "Ссылка",
        "Город",
        "Регион",
    ]
    assert df.loc[0, "Источник"] == "Avito"
    assert df.loc[0, "Компания"] == "ООО «Кофе»"
    assert df.loc[0, "Город"] == "Москва"
    assert df.loc[0, "ЗП от (тыс)"] == 72.0
    assert df.loc[0, "ЗП до (тыс)"] == 72.0
    assert df.loc[0, "Ставка в час (руб)"] == 400.0
    assert df.loc[0, "Средний доход за смену 12ч"] == 4800.0
    assert df.loc[0, "График"] == "2/2"
    assert df.loc[0, "Опыт"] == "без опыта"
    assert "Питание" in df.loc[0, "Условия/льготы"]
    assert df.loc[0, "Дата публикации"] == "2024-06-01"
    assert df.loc[0, "Дата сбора"] == "2024-06-03"
    assert "График=2/2" in df.loc[0, "Примечания расчета"]

    assert not logs.empty
    assert set(logs.columns) == {"stage", "message", "url", "timestamp"}
    assert logs.iloc[0]["stage"] in {"calc", "card"}


def test_build_analytics_tables():
    config = make_config()
    df, _ = normalize_records([SAMPLE_RECORD], config, collected_at=date(2024, 6, 3))
    analytics = build_analytics(df)

    assert analytics.total == 1
    assert isinstance(analytics.jobs, pd.DataFrame)
    assert not analytics.jobs.empty
    assert "Бариста" in analytics.jobs.iloc[0]["Должность"]
    assert isinstance(analytics.benefits, pd.DataFrame)
    assert "Питание" in analytics.benefits["Льгота"].tolist()
    assert analytics.hourly_share == 100.0
