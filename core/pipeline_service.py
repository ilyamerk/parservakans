from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from .schemas import ParserRunRequest, ParserRunResult

ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DIR = ROOT / "Exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

FETCH_SCRIPT = ROOT / "fetch_vacancies.py"
ANALYTICS_SCRIPT = ROOT / "build_job_analytics.py"
REPORT_SCRIPT = ROOT / "build_report_docx.py"

# Shared progress state (single-user; for multi-user use a dict keyed by run_id)
PROGRESS: dict = {
    "percent": 0,
    "message": "",
    "done": False,
    "error": None,
    "run_id": None,
    "row_count": 0,
}


class PipelineError(RuntimeError):
    pass


def _run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise PipelineError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _set_progress(percent: int, message: str) -> None:
    PROGRESS["percent"] = percent
    PROGRESS["message"] = message


def run_parser_pipeline(request: ParserRunRequest) -> ParserRunResult:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in request.query).strip("_") or "query"

    csv_path = EXPORTS_DIR / f"{safe_query}_{timestamp}_raw.csv"
    xlsx_path = EXPORTS_DIR / f"{safe_query}_{timestamp}_analytics.xlsx"
    docx_path = EXPORTS_DIR / f"{safe_query}_{timestamp}_report.docx"

    # Stage 1: Fetch vacancies (0-60%)
    _set_progress(5, "Инициализация сбора вакансий...")

    fetch_cmd = [
        sys.executable,
        str(FETCH_SCRIPT),
        "--query",
        request.query,
        "--area",
        str(request.area),
        "--city",
        request.city,
        "--pages",
        str(request.pages),
        "--per_page",
        str(request.per_page),
        "--pause",
        str(request.pause),
        "--search_in",
        request.search_in,
        "--workers",
        str(request.workers),
        "--timeout",
        str(request.timeout),
        "--role",
        request.role,
        "--out_csv",
        str(csv_path),
    ]
    if request.no_filter or not request.role:
        fetch_cmd.append("--no_filter")
    if request.avito_headful:
        fetch_cmd.append("--avito_headful")
    if request.avito_state:
        fetch_cmd.extend(["--avito_state", request.avito_state])

    _set_progress(10, "Собираю вакансии с HH.ru и Avito...")
    _run_cmd(fetch_cmd)

    # Count rows after fetch
    row_count_after_fetch = 0
    if csv_path.exists():
        row_count_after_fetch = len(pd.read_csv(csv_path))
    _set_progress(60, f"Собрано {row_count_after_fetch} вакансий. Строю Excel-аналитику...")

    # Stage 2: Build analytics (60-80%)
    _run_cmd([sys.executable, str(ANALYTICS_SCRIPT), "--input", str(csv_path), "--output", str(xlsx_path)])
    _set_progress(80, "Excel готов. Строю DOCX-отчёт...")

    # Stage 3: Build report (80-95%)
    _run_cmd(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            "--input_csv",
            str(csv_path),
            "--output_docx",
            str(docx_path),
            "--query",
            request.query,
            "--city",
            request.city,
        ]
    )
    _set_progress(95, "Финализация...")

    rows = row_count_after_fetch
    run_id = timestamp
    return ParserRunResult(run_id=run_id, csv_path=csv_path, xlsx_path=xlsx_path, docx_path=docx_path, row_count=rows)
