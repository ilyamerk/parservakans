import numpy as np
import pandas as pd

from build_job_analytics import RATES_SHEET_BASE, compute_metrics, write_excel


def test_compute_metrics_sets_dash_for_missing_shift_length():
    df = pd.DataFrame(
        [
            {
                "В час": 350.0,
                "Длительность смены": np.nan,
                "Средний совокупный доход при графике 2/2 по 12 часов": np.nan,
            }
        ]
    )
    result = compute_metrics(df)
    assert result.loc[0, "Длительность смены"] == "-"


def test_write_excel_creates_rates_sheet(tmp_path):
    df = pd.DataFrame([{"Должность": "Повар", "В час": 200.0, "Длительность смены": 12}])
    rates = [
        {"type": "hourly", "value": "200", "url": "https://example.com"},
        {"type": "shift", "value": "2400", "url": "https://example.com"},
    ]

    out_path = tmp_path / "result.xlsx"
    write_excel(df, out_path, rates=rates)

    workbook = pd.ExcelFile(out_path)
    assert RATES_SHEET_BASE in workbook.sheet_names

    rates_df = pd.read_excel(out_path, sheet_name=RATES_SHEET_BASE)
    assert list(rates_df.columns) == ["Ставка в час", "Ставка за смену", "Ссылка"]
    assert not rates_df.empty


def test_compute_metrics_uses_monthly_formula_with_schedule_mapping():
    df = pd.DataFrame(
        [
            {
                "ЗП от (т.р.)": 90.0,
                "В час": np.nan,
                "Длительность смены": 10.0,
                "График": "5/2",
                "Средний совокупный доход при графике 2/2 по 12 часов": np.nan,
            }
        ]
    )
    result = compute_metrics(df)
    assert result.loc[0, "В час"] == 409.09
    assert result.loc[0, "Средний совокупный доход при графике 2/2 по 12 часов"] == 4909.09


def test_compute_metrics_uses_default_22_shifts_for_unknown_schedule():
    df = pd.DataFrame(
        [
            {
                "ЗП от (т.р.)": 88.0,
                "В час": np.nan,
                "Длительность смены": 8.0,
                "График": "гибкий",
                "Средний совокупный доход при графике 2/2 по 12 часов": np.nan,
            }
        ]
    )
    result = compute_metrics(df)
    assert result.loc[0, "В час"] == 500.0
    assert result.loc[0, "Средний совокупный доход при графике 2/2 по 12 часов"] == 6000.0


def test_write_excel_strips_illegal_xml_control_chars(tmp_path):
    bad_text = "Повар\x0bна смену"
на смену"
    df = pd.DataFrame([{"Должность": bad_text, "Ссылка": "https://example.com"}])

    out_path = tmp_path / "illegal_chars.xlsx"
    write_excel(df, out_path)

    restored = pd.read_excel(out_path, sheet_name="Данные")
    assert restored.loc[0, "Должность"] == "Поварна смену"
