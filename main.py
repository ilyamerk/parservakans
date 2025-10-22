# main.py — пайплайн: сбор → Excel → DOCX (ПК-Avito)
import argparse, subprocess, sys, shlex
from pathlib import Path
import urllib.parse
from urllib.parse import urljoin, urlsplit
import re, json, time

GR_DEBUG = True  # временно: сохраняем HTML страниц GR в Exports/_debug


ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "Exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    ROOT / "fetch_vacancies.py",
    ROOT / "parsers" / "fetch_vacancies.py",
]
FETCH = next((p for p in CANDIDATES if p.exists()), None)
if not FETCH:
    raise SystemExit("Не найден fetch_vacancies.py")

ANALYTICS = ROOT / "build_job_analytics.py"
REPORT    = ROOT / "build_report_docx.py"

def run(cmd, cwd=None):
    print("▶", " ".join(shlex.quote(str(x)) for x in cmd))
    subprocess.run(cmd, check=True, cwd=cwd)

def main():
    ap = argparse.ArgumentParser(description="Полный пайплайн: сбор → аналитика → DOCX")
    ap.add_argument("--query", default="Бариста")
    ap.add_argument("--area", default="1")
    ap.add_argument("--city", default="Москва")
    ap.add_argument("--pages", default="3")
    ap.add_argument("--per_page", default="50")
    ap.add_argument("--pause", default="0.9")
    ap.add_argument("--search_in", default="name")
    ap.add_argument("--workers", default="8")
    ap.add_argument("--timeout", default="8.0")
    ap.add_argument("--avito_headful", action="store_true")
    ap.add_argument("--avito_state", default="avito_state.json")
    ap.add_argument("--role", default="")
    ap.add_argument("--no_filter", action="store_true")

    args = ap.parse_args()

    RAW_CSV = EXPORT_DIR / f"{args.query}_raw.csv"

    fetch_cmd = [
        sys.executable, str(FETCH),
        "--query", args.query,
        "--area", str(args.area),
        "--city", args.city,
        "--pages", str(args.pages),
        "--per_page", str(args.per_page),
        "--pause", str(args.pause),
        "--search_in", args.search_in,
        "--workers", str(args.workers),
        "--timeout", str(args.timeout),
        "--role", args.role,
        "--out_csv", str(RAW_CSV),
    ]
    if getattr(args, "no_filter", False) or args.role == "":
        fetch_cmd.append("--no_filter")
    if getattr(args, "avito_headful", False):
        fetch_cmd.append("--avito_headful")
    if getattr(args, "avito_state", None):
        fetch_cmd += ["--avito_state", args.avito_state]

    print("\n=== Шаг 1/3: сбор вакансий (HH/Avito) ===")
    run(fetch_cmd, cwd=ROOT)
    print(f"RAW CSV готов: {RAW_CSV}")

    print("\n=== Шаг 2/3: аналитика в Excel ===")
    AN_XLSX = EXPORT_DIR / "Аналитика.xlsx"
    run([sys.executable, str(ANALYTICS), "--input", str(RAW_CSV), "--output", str(AN_XLSX)], cwd=ROOT)
    print(f"Excel готов: {AN_XLSX}")

    print("\n=== Шаг 3/3: DOCX-отчёт ===")
    DOCX = EXPORT_DIR / "Отчёт.docx"
    run([sys.executable, str(REPORT), "--input_csv", str(RAW_CSV), "--output_docx", str(DOCX),
         "--query", args.query, "--city", args.city], cwd=ROOT)
    print(f"DOCX готов: {DOCX}")

    print("\nГотово.")

if __name__ == "__main__":
    main()
