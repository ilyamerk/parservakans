import unittest

from fetch_vacancies import (
    is_shift_rate,
    extract_shift_rate,
    collect_shift_rate_rows,
    compute_hourly_rate,
    apply_computed_hourly_rates,
)
from build_job_analytics import write_excel
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd


class ShiftRateDetectionTests(unittest.TestCase):
    def test_is_shift_rate_positive(self):
        text = "Смена 12ч — 3000 ₽"
        self.assertTrue(is_shift_rate(text))

    def test_is_shift_rate_per_shift(self):
        text = "$150 per shift guaranteed"
        self.assertTrue(is_shift_rate(text))

    def test_is_shift_rate_negative_by_hint(self):
        text = "оклад 60 000 ₽/мес"
        self.assertFalse(is_shift_rate(text))

    def test_is_shift_rate_negative_day(self):
        text = "в день 2000 ₽"
        self.assertFalse(is_shift_rate(text))


class ExtractShiftRateTests(unittest.TestCase):
    def test_extract_simple_shift_rate(self):
        data = extract_shift_rate("Смена 12ч — 3000 ₽")
        self.assertIsNotNone(data)
        self.assertEqual(data["value"], "3000 ₽")
        self.assertEqual(data["min"], 3000.0)
        self.assertEqual(data["max"], 3000.0)
        self.assertEqual(data["currency"], "RUB")

    def test_extract_shift_rate_without_currency(self):
        data = extract_shift_rate("Оплата 1800/смена")
        self.assertIsNotNone(data)
        self.assertEqual(data["value"], "1800 ₽")
        self.assertEqual(data["min"], 1800.0)
        self.assertEqual(data["max"], 1800.0)
        self.assertEqual(data["currency"], "RUB")

    def test_extract_shift_rate_dollars(self):
        data = extract_shift_rate("$50-60 per shift")
        self.assertIsNotNone(data)
        self.assertEqual(data["value"], "$50–60")
        self.assertEqual(data["min"], 50.0)
        self.assertEqual(data["max"], 60.0)
        self.assertEqual(data["currency"], "USD")

    def test_extract_shift_rate_euro(self):
        data = extract_shift_rate("€40 per shift")
        self.assertIsNotNone(data)
        self.assertEqual(data["value"], "€40")
        self.assertEqual(data["currency"], "EUR")


class CollectShiftRateRowsTests(unittest.TestCase):
    def test_collect_shift_rows(self):
        rows = [
            {
                "Ссылка": "https://example.com/job",
                "__rate_text": "Оплата 1800/смена или 200/час",
            }
        ]
        rates = collect_shift_rate_rows(rows)
        self.assertEqual(len(rates), 1)
        self.assertEqual(rates[0]["value"], "1800 ₽")
        self.assertEqual(rates[0]["url"], "https://example.com/job")


class HourlyRateComputationTests(unittest.TestCase):
    def test_compute_hourly_rate_with_explicit_fields(self):
        vac = {
            "ЗП от (т.р.)": 120,
            "График": "5/2",
            "Длительность смены": "8 часов",
        }
        rate = compute_hourly_rate(vac)
        self.assertAlmostEqual(rate, 652.2)

    def test_compute_hourly_rate_with_shift_map(self):
        vac = {
            "ЗП от (т.р.)": 150,
            "График": "сутки/двое",
            "Длительность смены": "24",
        }
        rate = compute_hourly_rate(vac)
        self.assertAlmostEqual(rate, 625.0)

    def test_compute_hourly_rate_defaults_hours(self):
        vac = {
            "ЗП от (т.р.)": 90,
            "График": "3/3",
        }
        rate = compute_hourly_rate(vac)
        self.assertAlmostEqual(rate, 500.0)

    def test_compute_hourly_rate_from_text_schedule(self):
        vac = {
            "ЗП от (т.р.)": "120",
            "Описание": "График 2 через 2, смена 12 ч",
        }
        rate = compute_hourly_rate(vac)
        self.assertAlmostEqual(rate, 666.7)

    def test_compute_hourly_rate_requires_salary_from(self):
        vac = {
            "ЗП от (т.р.)": None,
            "ЗП до (т.р.)": 200,
            "График": "2/2",
        }
        self.assertIsNone(compute_hourly_rate(vac))

    def test_apply_computed_hourly_rates_populates_fields(self):
        rows = [
            {
                "ЗП от (т.р.)": 180,
                "График": "2/2",
                "Длительность смены": "12",
                "В час": "",
            }
        ]
        apply_computed_hourly_rates(rows)
        self.assertEqual(rows[0]["В час"], 1000.0)
        self.assertEqual(rows[0]["Ставка (расчётная) в час, ₽"], 1000.0)

    def test_apply_computed_hourly_rates_preserves_existing_hourly(self):
        rows = [
            {
                "ЗП от (т.р.)": 120,
                "График": "5/2",
                "Длительность смены": "8",
                "В час": 700.0,
            }
        ]
        apply_computed_hourly_rates(rows)
        self.assertEqual(rows[0]["В час"], 700.0)
        self.assertAlmostEqual(rows[0]["Ставка (расчётная) в час, ₽"], 652.2)


class ExportShiftSheetTests(unittest.TestCase):
    def test_write_excel_creates_shift_sheet(self):
        df = pd.DataFrame([
            {"Должность": "Повар", "Ссылка": "https://example.com/job"}
        ])
        rates = [
            {"value": "1800 ₽", "url": "https://example.com/job", "raw": "1800/смена"}
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.xlsx"
            write_excel(df, path, rates=rates)
            self.assertTrue(path.exists())
            xls = pd.ExcelFile(path)
            self.assertIn("Ставки за смену", xls.sheet_names)
            data = pd.read_excel(path, sheet_name="Ставки за смену")
            self.assertListEqual(list(data.columns), ["Ставка за смену", "Ссылка"])
            self.assertEqual(data.iloc[0]["Ставка за смену"], "1800 ₽")
            self.assertEqual(data.iloc[0]["Ссылка"], "https://example.com/job")


if __name__ == "__main__":
    unittest.main()
