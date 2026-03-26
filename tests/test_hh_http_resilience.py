import requests

from fetch_vacancies import HHHTTPConfig, _request_json_with_retries, hh_search


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class _SeqSession:
    def __init__(self, seq):
        self.seq = list(seq)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        item = self.seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_retry_on_connect_timeout_then_success(monkeypatch):
    sess = _SeqSession(
        [
            requests.exceptions.ConnectTimeout("timed out"),
            _Resp(200, {"items": [{"id": "1"}], "pages": 1}),
        ]
    )
    monkeypatch.setattr("fetch_vacancies.time.sleep", lambda *_: None)
    payload, attempts, err = _request_json_with_retries(
        sess,
        "https://api.hh.ru/vacancies",
        params={"page": 0},
        config=HHHTTPConfig(max_retries=3, backoff_factor=0),
        stage="поиск вакансий",
    )
    assert payload is not None
    assert payload["items"][0]["id"] == "1"
    assert attempts == 2
    assert err is None


def test_retry_on_503_then_success(monkeypatch):
    sess = _SeqSession(
        [
            _Resp(503, {}),
            _Resp(200, {"items": [{"id": "ok"}], "pages": 1}),
        ]
    )
    monkeypatch.setattr("fetch_vacancies.time.sleep", lambda *_: None)
    payload, attempts, err = _request_json_with_retries(
        sess,
        "https://api.hh.ru/vacancies",
        params={"page": 0},
        config=HHHTTPConfig(max_retries=3, backoff_factor=0),
        stage="поиск вакансий",
    )
    assert payload is not None
    assert attempts == 2
    assert err is None


def test_hh_search_handles_page_failure_without_crash(monkeypatch):
    class _SearchSession:
        def __init__(self):
            self.n = 0

        def get(self, url, params=None, timeout=None):
            self.n += 1
            page = params["page"]
            if page == 0:
                return _Resp(200, {"items": [{"id": "p0"}], "pages": 3})
            if page == 1:
                raise requests.exceptions.ConnectTimeout("timed out")
            return _Resp(200, {"items": [{"id": "p2"}], "pages": 3})

    monkeypatch.setattr("fetch_vacancies._get_sess", lambda: _SearchSession())
    monkeypatch.setattr("fetch_vacancies.time.sleep", lambda *_: None)

    items = hh_search("бариста", pages=3, pause=0, timeout=10, config=HHHTTPConfig(max_retries=2, backoff_factor=0, pause_between_requests=0))
    assert [i["id"] for i in items] == ["p0", "p2"]


def test_hh_search_raises_clear_error_when_api_unavailable(monkeypatch):
    class _BadSession:
        def get(self, *args, **kwargs):
            raise requests.exceptions.ConnectTimeout("timed out")

    monkeypatch.setattr("fetch_vacancies._get_sess", lambda: _BadSession())
    monkeypatch.setattr("fetch_vacancies.time.sleep", lambda *_: None)

    try:
        hh_search("бариста", pages=2, pause=0, config=HHHTTPConfig(max_retries=2, backoff_factor=0, pause_between_requests=0))
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "HH API временно недоступен" in str(exc)
