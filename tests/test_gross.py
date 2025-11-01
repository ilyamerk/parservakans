import unittest

from fetch_vacancies import is_gross_salary


class GrossSalaryTests(unittest.TestCase):
    def test_do_vycheta_ndfl(self):
        text = "ЗП 120 000–150 000 руб. (до вычета НДФЛ)"
        self.assertTrue(is_gross_salary(text))

    def test_na_ruki(self):
        text = "150 000 руб. на руки"
        self.assertFalse(is_gross_salary(text))

    def test_gross_keyword(self):
        text = "Оклад gross 180 000"
        self.assertTrue(is_gross_salary(text))

    def test_brutto(self):
        text = "Брутто 200 000 + бонус"
        self.assertTrue(is_gross_salary(text))

    def test_net_keyword(self):
        text = "Net 140 000"
        self.assertFalse(is_gross_salary(text))


if __name__ == "__main__":
    unittest.main()
