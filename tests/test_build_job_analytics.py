import pandas as pd

from build_job_analytics import write_excel, RATES_SHEET_BASE


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
