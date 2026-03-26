from __future__ import annotations

import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core.pipeline_service import PipelineError, run_parser_pipeline, PROGRESS
from core.schemas import ParserRunRequest

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="Parser Vakans")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))

RUNS: dict[str, dict[str, Any]] = {}


def _preview_from_csv(csv_path: Path, limit: int = 20) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path).head(limit).fillna("")
    return df.to_dict(orient="records")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/run")
def run_from_form(
    query: str = Form("Бариста"),
    area: int = Form(1),
    city: str = Form("Москва"),
    pages: int = Form(3),
    per_page: int = Form(50),
    pause: float = Form(0.9),
    search_in: str = Form("name"),
    workers: int = Form(8),
    timeout: float = Form(8.0),
    role: str = Form(""),
    no_filter: bool = Form(False),
):
    req = ParserRunRequest(
        query=query,
        area=area,
        city=city,
        pages=pages,
        per_page=per_page,
        pause=pause,
        search_in=search_in,
        workers=workers,
        timeout=timeout,
        role=role,
        no_filter=no_filter,
    )

    # Reset progress
    PROGRESS["percent"] = 0
    PROGRESS["message"] = "Инициализация..."
    PROGRESS["done"] = False
    PROGRESS["error"] = None
    PROGRESS["run_id"] = None
    PROGRESS["row_count"] = 0

    def run_in_bg():
        try:
            result = run_parser_pipeline(req)
            RUNS[result.run_id] = {
                "result": asdict(result),
                "request": asdict(req),
            }
            PROGRESS["run_id"] = result.run_id
            PROGRESS["row_count"] = result.row_count
            PROGRESS["percent"] = 100
            PROGRESS["message"] = f"Готово! Найдено вакансий: {result.row_count}"
            PROGRESS["done"] = True
        except PipelineError as exc:
            PROGRESS["percent"] = 100
            PROGRESS["message"] = "Ошибка"
            PROGRESS["done"] = True
            PROGRESS["error"] = str(exc)

    thread = threading.Thread(target=run_in_bg, daemon=True)
    thread.start()

    return JSONResponse({"status": "started", "run_id": "pending"})


@app.get("/progress")
def get_progress():
    return JSONResponse({
        "percent": PROGRESS["percent"],
        "message": PROGRESS["message"],
        "done": PROGRESS["done"],
        "error": PROGRESS["error"],
        "run_id": PROGRESS.get("run_id"),
        "row_count": PROGRESS.get("row_count", 0),
    })


@app.post("/api/run")
def run_api(payload: ParserRunRequest):
    try:
        result = run_parser_pipeline(payload)
    except PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    RUNS[result.run_id] = {"result": asdict(result), "request": asdict(payload)}
    return {"run_id": result.run_id, "row_count": result.row_count}


@app.get("/api/result/{run_id}")
def result_api(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run_id not found")
    result = run["result"].copy()
    result["preview"] = _preview_from_csv(Path(result["csv_path"]))
    return JSONResponse(result)


@app.get("/download/{run_id}/{fmt}")
def download(run_id: str, fmt: str):
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run_id not found")

    result = run["result"]
    if fmt == "csv":
        path = Path(result["csv_path"])
        media = "text/csv"
    elif fmt == "xlsx":
        path = Path(result["xlsx_path"])
        media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif fmt == "docx":
        path = Path(result["docx_path"])
        media = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        raise HTTPException(status_code=400, detail="unsupported format")

    if not path.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path=path, filename=path.name, media_type=media)
