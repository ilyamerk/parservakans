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


def run_parser_pipeline(request: ParserRunRequest) -> ParserRunResult:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in request.query).strip("_") or "query"

    csv_path = EXPORTS_DIR / f"{safe_query}_{timestamp}_raw.csv"
    xlsx_path = EXPORTS_DIR / f"{safe_query}_{timestamp}_analytics.xlsx"
    docx_path = EXPORTS_DIR / f"{safe_query}_{timestamp}_report.docx"

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

    _run_cmd(fetch_cmd)
    _run_cmd([sys.executable, str(ANALYTICS_SCRIPT), "--input", str(csv_path), "--output", str(xlsx_path)])
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

    rows = len(pd.read_csv(csv_path)) if csv_path.exists() else 0
    run_id = timestamp
    return ParserRunResult(run_id=run_id, csv_path=csv_path, xlsx_path=xlsx_path, docx_path=docx_path, row_count=rows)
