import pandas as pd

from fetch_vacancies import main as fetch_main


def test_hh_smoke_run_with_mocked_search(monkeypatch, tmp_path):
    out_csv = tmp_path / "hh_raw.csv"

    monkeypatch.setattr(
        "fetch_vacancies.hh_search",
        lambda *args, **kwargs: [
            {
                "id": "1",
                "name": "Бариста",
                "employer": {"name": "Coffee LLC"},
                "published_at": "2025-01-01",
                "salary": {"from": 80000, "to": 100000, "currency": "RUR"},
                "experience": {"name": "Нет опыта"},
                "employment": {"name": "Полная занятость"},
                "schedule": {"name": "2/2"},
                "snippet": {"responsibility": "Готовить кофе", "requirement": "Общение с гостями"},
                "alternate_url": "https://hh.ru/vacancy/1",
            }
        ],
    )

    monkeypatch.setattr("fetch_vacancies.hh_details", lambda vid: {})
    monkeypatch.setattr("fetch_vacancies._extract_schedule_from_html", lambda url: (None, None))

    monkeypatch.setattr(
        "sys.argv",
        [
            "fetch_vacancies.py",
            "--query", "бариста",
            "--out_csv", str(out_csv),
            "--no_filter",
            "--no_gr",
            "--no_avito",
        ],
    )

    fetch_main()

    df = pd.read_csv(out_csv)
    assert len(df) >= 1
    assert "Источник" in df.columns
