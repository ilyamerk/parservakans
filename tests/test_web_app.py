from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from core.schemas import ParserRunResult


client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_api_run_and_result(monkeypatch, tmp_path):
    csv_path = tmp_path / "out.csv"
    csv_path.write_text("col1,col2\n1,2\n", encoding="utf-8")
    xlsx_path = tmp_path / "out.xlsx"
    xlsx_path.write_bytes(b"xlsx")
    docx_path = tmp_path / "out.docx"
    docx_path.write_bytes(b"docx")

    def fake_run(_request):
        return ParserRunResult(
            run_id="run123",
            csv_path=Path(csv_path),
            xlsx_path=Path(xlsx_path),
            docx_path=Path(docx_path),
            row_count=1,
        )

    monkeypatch.setattr("app.main.run_parser_pipeline", fake_run)

    payload = {"query": "Бариста", "city": "Москва"}
    run_resp = client.post("/api/run", json=payload)
    assert run_resp.status_code == 200
    assert run_resp.json()["run_id"] == "run123"

    result_resp = client.get("/api/result/run123")
    assert result_resp.status_code == 200
    data = result_resp.json()
    assert data["row_count"] == 1
    assert data["preview"][0]["col1"] == 1

    download_resp = client.get("/download/run123/csv")
    assert download_resp.status_code == 200
