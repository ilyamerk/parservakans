from fetch_vacancies import hh_search


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "timeout": timeout})
        payload = {
            "items": [{"id": f"id-{params.get('page', 0)}"}],
            "pages": 100,
        }
        return _FakeResponse(payload)


def test_hh_search_supports_start_page(monkeypatch):
    fake_session = _FakeSession()

    monkeypatch.setattr("fetch_vacancies._get_sess", lambda: fake_session)
    monkeypatch.setattr("fetch_vacancies.time.sleep", lambda *_: None)

    items = hh_search(
        query="бариста",
        area=1,
        pages=2,
        per_page=50,
        pause=0,
        search_in="name",
        start_page=3,
    )

    assert [call["params"]["page"] for call in fake_session.calls] == [3, 4]
    assert [item["id"] for item in items] == ["id-3", "id-4"]
