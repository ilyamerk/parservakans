import pandas as pd
import numpy as np

from build_job_analytics import compute_metrics, write_excel, RATES_SHEET_BASE


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
    df = pd.DataFrame(
        [
            {
                "Должность": "Повар",
                "В час": 200.0,
                "Длительность смены": 12,
            }
        ]
    )
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


def test_write_excel_strips_illegal_xml_control_chars(tmp_path):
    bad_text = "Поварна смену"
    df = pd.DataFrame([{"Должность": bad_text, "Ссылка": "https://example.com"}])

    out_path = tmp_path / "illegal_chars.xlsx"
    write_excel(df, out_path)

    restored = pd.read_excel(out_path, sheet_name="Данные")
    assert restored.loc[0, "Должность"] == "Поварна смену"
