"""Diagnostic tests explaining why the HH parser returns 0 vacancies.

Root cause summary
------------------
`hh_search` calls ``api.hh.ru/vacancies`` through a shared ``requests.Session``
configured with a browser ``User-Agent`` ("Mozilla/5.0 ... Chrome/...").
The HH public API enforces its own UA policy and responds with **403
Forbidden** when it sees a generic browser UA (the required format is
``AppName/Version (contact)``).

In the current code the failure is silent::

    if r.status_code != 200:
        break

No message is printed and no exception is raised, so `hh_search` returns an
empty list, `map_hh` gets nothing to map, and the pipeline writes 0 rows.
That matches the "parser returns 0 vacancies" symptom reported by the user.

The retry policy configured in ``_get_sess()`` does **not** cover 403:
``status_forcelist=[429, 500, 502, 503, 504]``. So a 403 is not retried and
not escalated.

The tests below lock down this behaviour and make future regressions loud:
* the session used for HH must advertise an HH-compliant User-Agent;
* non-200 responses must be logged with status code and a response snippet;
* empty ``items`` lists must not crash the pagination loop.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import pytest

import fetch_vacancies
from fetch_vacancies import hh_search


class _FakeResponse:
    def __init__(self, payload: Any = None, status_code: int = 200, text: str = ""):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code
        self.text = text or ""

    def json(self):
        return self._payload


class _RecordingSession:
    """Minimal stand-in for ``requests.Session`` used by the parser."""

    def __init__(self, responses: List[_FakeResponse]):
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []
        self.headers: Dict[str, str] = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "timeout": timeout})
        if self._responses:
            return self._responses.pop(0)
        return _FakeResponse({"items": [], "pages": 0})


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("fetch_vacancies.time.sleep", lambda *_: None)


def _install_session(monkeypatch, session):
    monkeypatch.setattr("fetch_vacancies._get_sess", lambda: session)


# ---------------------------------------------------------------------------
# 1. The actual "0 vacancies" symptom: 403 is silently swallowed.
# ---------------------------------------------------------------------------

def test_hh_search_returns_empty_when_api_responds_403(monkeypatch):
    """Reproduces the user-visible symptom: HH returns 403, parser returns []."""
    session = _RecordingSession([
        _FakeResponse(status_code=403, text="Forbidden: bad User-Agent"),
    ])
    _install_session(monkeypatch, session)

    items = hh_search(
        query="бариста",
        area=1,
        pages=3,
        per_page=50,
        pause=0,
        search_in="name",
    )

    assert items == []
    # Only the first page was attempted before breaking.
    assert len(session.calls) == 1


def test_hh_search_logs_status_and_body_on_non_200(monkeypatch, capsys):
    """Regression guard: non-200 responses must be logged so 0 vacancies is explainable.

    Without a log, a 403/401/404 from HH is indistinguishable from an empty
    query result. Operators get "0 vacancies" with no clue why.
    """
    session = _RecordingSession([
        _FakeResponse(status_code=403, text="Forbidden: User-Agent is missing or invalid"),
    ])
    _install_session(monkeypatch, session)

    hh_search(
        query="бариста",
        area=1,
        pages=1,
        per_page=50,
        pause=0,
        search_in="name",
    )

    out = capsys.readouterr().out
    assert "403" in out, f"Expected HTTP 403 mentioned in log, got: {out!r}"
    assert re.search(r"\[HH\]", out), f"Expected [HH] tagged log, got: {out!r}"


# ---------------------------------------------------------------------------
# 2. The upstream cause: a browser UA. HH API rejects it.
# ---------------------------------------------------------------------------

def test_hh_session_user_agent_is_hh_api_compliant():
    """HH API requires a UA like ``AppName/Version (contact)``.

    A plain ``Mozilla/5.0 ... Chrome/...`` is treated as a browser scraper and
    returns 403. We assert the UA advertised by the session used for HH calls
    matches HH's format and does not look like a stock browser.
    """
    # Reset the cached session so the test sees the current module config.
    fetch_vacancies._SESS = None
    sess = fetch_vacancies._get_sess()
    ua = sess.headers.get("User-Agent", "")

    assert ua, "User-Agent must be set for HH API requests"
    assert "Mozilla/" not in ua and "Chrome/" not in ua, (
        f"HH API rejects browser-style User-Agent; got: {ua!r}"
    )
    # Format: AppName/Version (contact) — e.g. parservakans/1.0 (https://...)
    assert re.match(r"^[\w.\-]+/[\w.\-]+\s*\(.+\)\s*$", ua), (
        f"User-Agent must follow 'AppName/Version (contact)' per HH API docs; got: {ua!r}"
    )


# ---------------------------------------------------------------------------
# 3. Sanity: the pagination loop does not misbehave on empty pages.
# ---------------------------------------------------------------------------

def test_hh_search_stops_cleanly_on_empty_result(monkeypatch):
    session = _RecordingSession([
        _FakeResponse({"items": [], "pages": 0}),
    ])
    _install_session(monkeypatch, session)

    items = hh_search(
        query="не существует никогда",
        area=1,
        pages=3,
        per_page=50,
        pause=0,
        search_in="name",
    )

    assert items == []
    # ``pages: 0`` means "no results" → stop after the first request.
    assert len(session.calls) == 1


def test_hh_search_collects_items_across_pages(monkeypatch):
    session = _RecordingSession([
        _FakeResponse({"items": [{"id": "a"}, {"id": "b"}], "pages": 3}),
        _FakeResponse({"items": [{"id": "c"}], "pages": 3}),
        _FakeResponse({"items": [{"id": "d"}], "pages": 3}),
    ])
    _install_session(monkeypatch, session)

    items = hh_search(
        query="бариста",
        area=1,
        pages=3,
        per_page=50,
        pause=0,
        search_in="name",
    )

    assert [i["id"] for i in items] == ["a", "b", "c", "d"]
    assert [c["params"]["page"] for c in session.calls] == [0, 1, 2]


def test_hh_search_sends_search_field_parameter(monkeypatch):
    session = _RecordingSession([
        _FakeResponse({"items": [{"id": "x"}], "pages": 1}),
    ])
    _install_session(monkeypatch, session)

    hh_search(
        query="бариста",
        area=1,
        pages=1,
        per_page=50,
        pause=0,
        search_in="name",
    )

    assert session.calls[0]["params"].get("search_field") == "name"
    assert session.calls[0]["params"]["text"] == "бариста"
    assert session.calls[0]["params"]["area"] == 1
