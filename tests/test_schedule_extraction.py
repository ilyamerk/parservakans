from fetch_vacancies import extract_schedule_strict, _sanitize_schedule


def test_extract_schedule_strict_collects_multiple_patterns():
    text = "График 2-2 или пять через два, возможен сутки через трое"
    assert extract_schedule_strict(text) == "2/2, 5/2, 1/3"


def test_extract_schedule_strict_handles_text_variants():
    assert extract_schedule_strict("Работа: два через два") == "2/2"
    assert extract_schedule_strict("Условия: шесть через один") == "6/1"
    assert extract_schedule_strict("Условия: четыре через три") == "4/3"
    assert extract_schedule_strict("График: день через день") == "1/1"


def test_extract_schedule_strict_does_not_map_generic_schedule_names():
    assert extract_schedule_strict("", sched_src="Полная занятость") is None
    assert extract_schedule_strict("", sched_src="Полный день") is None


def test_sanitize_schedule_accepts_only_numeric_x_slash_y():
    assert _sanitize_schedule("2/2, 5/2") == "2/2, 5/2"
    assert _sanitize_schedule("гибкий") is None
