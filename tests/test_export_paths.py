from pathlib import Path

from fetch_vacancies import _resolve_export_paths


def test_resolve_export_paths_for_csv_target():
    csv_path, xlsx_path = _resolve_export_paths("Exports/Бариста_raw.csv")
    assert csv_path == Path("Exports/Бариста_raw.csv")
    assert xlsx_path == Path("Exports/Бариста_raw_view.xlsx")


def test_resolve_export_paths_for_xlsx_target():
    csv_path, xlsx_path = _resolve_export_paths("Exports/Бариста_raw_view.xlsx")
    assert csv_path == Path("Exports/Бариста_raw_view.csv")
    assert xlsx_path == Path("Exports/Бариста_raw_view.xlsx")

