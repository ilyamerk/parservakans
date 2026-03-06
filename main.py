# main.py — пайплайн: сбор hh.ru → Excel → DOCX
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "Exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

FETCH = ROOT / "fetch_vacancies.py"
ANALYTICS = ROOT / "build_job_analytics.py"
REPORT = ROOT / "build_report_docx.py"


def run(cmd, cwd=None):
    print("▶", " ".join(shlex.quote(str(x)) for x in cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def main():
    ap = argparse.ArgumentParser(description="Полный HH-пайплайн: сбор → аналитика → DOCX")
    ap.add_argument("--query", default="Бариста")
    ap.add_argument("--area", default="1")
    ap.add_argument("--city", default="Москва")
    ap.add_argument("--pages", default="3")
    ap.add_argument("--per_page", default="50")
    ap.add_argument("--pause", default="0.9")
    ap.add_argument("--search_in", default="name")
    ap.add_argument("--workers", default="8")
    ap.add_argument("--timeout", default="8.0")
    ap.add_argument("--role", default="")
    ap.add_argument("--no_filter", action="store_true")
    args = ap.parse_args()

    raw_csv = EXPORT_DIR / f"{args.query}_raw.csv"
    fetch_cmd = [
        sys.executable,
        str(FETCH),
        "--query",
        args.query,
        "--area",
        str(args.area),
        "--city",
        args.city,
        "--pages",
        str(args.pages),
        "--per_page",
        str(args.per_page),
        "--pause",
        str(args.pause),
        "--search_in",
        args.search_in,
        "--workers",
        str(args.workers),
        "--timeout",
        str(args.timeout),
        "--role",
        args.role,
        "--out_csv",
        str(raw_csv),
    ]
    if getattr(args, "no_filter", False) or args.role == "":
        fetch_cmd.append("--no_filter")

    print("\n=== Шаг 1/3: сбор вакансий (HH) ===")
    run(fetch_cmd, cwd=ROOT)

    an_xlsx = EXPORT_DIR / "Аналитика.xlsx"
    print("\n=== Шаг 2/3: аналитика в Excel ===")
    run([sys.executable, str(ANALYTICS), "--input", str(raw_csv), "--output", str(an_xlsx)], cwd=ROOT)

    docx = EXPORT_DIR / "Отчёт.docx"
    print("\n=== Шаг 3/3: DOCX-отчёт ===")
    run([
        sys.executable,
        str(REPORT),
        "--input_csv",
        str(raw_csv),
        "--output_docx",
        str(docx),
        "--query",
        args.query,
        "--city",
        args.city,
    ], cwd=ROOT)


if __name__ == "__main__":
    main()
