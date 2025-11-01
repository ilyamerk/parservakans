import unittest

from fetch_vacancies import (
    is_hourly_rate,
    is_shift_rate,
    extract_rate,
    collect_rate_rows,
    filter_rows_without_shift_rates,
)


class HourlyShiftDetectionTests(unittest.TestCase):
    def test_is_hourly_rate_positive(self):
        text = "Оплата 1200 руб/час"
        self.assertTrue(is_hourly_rate(text))

    def test_is_hourly_rate_negative(self):
        text = "Оклад 60 000 ₽ в месяц"
        self.assertFalse(is_hourly_rate(text))

    def test_is_shift_rate_positive(self):
        text = "Смена 12ч — 3000 ₽"
        self.assertTrue(is_shift_rate(text))

    def test_is_shift_rate_per_shift(self):
        text = "$150 per shift guaranteed"
        self.assertTrue(is_shift_rate(text))


class ExtractRateTests(unittest.TestCase):
    def test_extract_single_with_suffix(self):
        data = extract_rate("1200 руб/час + %")
        self.assertIsNotNone(data)
        self.assertEqual(data["value"], "1200 ₽ + %")
        self.assertEqual(data["min"], 1200.0)
        self.assertEqual(data["max"], 1200.0)
        self.assertEqual(data["currency"], "RUB")

    def test_extract_range_suffix_currency(self):
        data = extract_rate("1500-2000 руб/ч")
        self.assertIsNotNone(data)
        self.assertEqual(data["value"], "1500–2000 ₽")
        self.assertEqual(data["min"], 1500.0)
        self.assertEqual(data["max"], 2000.0)
        self.assertEqual(data["currency"], "RUB")

    def test_extract_range_prefix_currency(self):
        data = extract_rate("$15-18 per hour")
        self.assertIsNotNone(data)
        self.assertEqual(data["value"], "$15–18")
        self.assertEqual(data["min"], 15.0)
        self.assertEqual(data["max"], 18.0)
        self.assertEqual(data["currency"], "USD")

    def test_extract_shift_phrase(self):
        data = extract_rate("от 1 200 до 1 600 за смену")
        self.assertIsNotNone(data)
        self.assertEqual(data["value"], "от 1 200 до 1 600")
        self.assertEqual(data["min"], 1200.0)
        self.assertEqual(data["max"], 1600.0)
        self.assertIsNone(data["currency"])

    def test_extract_euro_hour(self):
        data = extract_rate("€12/hour")
        self.assertIsNotNone(data)
        self.assertEqual(data["value"], "€12")
        self.assertEqual(data["min"], 12.0)
        self.assertEqual(data["max"], 12.0)
        self.assertEqual(data["currency"], "EUR")


class CollectRateRowsTests(unittest.TestCase):
    def test_collect_both_hourly_and_shift(self):
        row = {
            "Ссылка": "https://example.com/job",
            "__rate_text": "Оплата 1800/смена или 200/час",
        }
        rates = collect_rate_rows([row])
        self.assertEqual(len(rates), 2)
        types = {r["type"] for r in rates}
        self.assertEqual(types, {"hourly", "shift"})
        hourly = next(r for r in rates if r["type"] == "hourly")
        shift = next(r for r in rates if r["type"] == "shift")
        self.assertEqual(hourly["value"], "200")
        self.assertEqual(shift["value"], "1800")
        self.assertEqual(hourly["url"], row["Ссылка"])
        self.assertEqual(shift["url"], row["Ссылка"])


class FilterShiftRowsTests(unittest.TestCase):
    def test_filter_excludes_shift_rows_from_main(self):
        rows = [
            {"Ссылка": "https://example.com/a"},
            {"Ссылка": "https://example.com/b"},
        ]
        rate_rows = [
            {"type": "shift", "url": "https://example.com/a"},
            {"type": "hourly", "url": "https://example.com/b"},
        ]
        filtered = filter_rows_without_shift_rates(rows, rate_rows)
        urls = {row["Ссылка"] for row in filtered}
        self.assertNotIn("https://example.com/a", urls)
        self.assertIn("https://example.com/b", urls)


if __name__ == "__main__":
    unittest.main()
