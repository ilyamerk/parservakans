#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def parse_legacy_args(argv):
    if "--output" not in argv:
        print("Usage: run_pipeline.py --query ... --area ... [--city CITY] --pages ... --per_page ... --output OUT.xlsx")
        sys.exit(1)
    args = argv[:]
    out_idx = args.index("--output")
    out_val = args[out_idx + 1]
    args_wo_output = args[:out_idx] + args[out_idx + 2 :]

    def get_opt(name, default=""):
        if name in args_wo_output:
            i = args_wo_output.index(name)
            return args_wo_output[i + 1] if i + 1 < len(args_wo_output) else default
        return default

    return out_val, args_wo_output, get_opt("--query", ""), get_opt("--city", "")


def run_pipeline(out: str, args_wo_output: list[str], query_text: str, city_text: str) -> None:
    tmp_csv = Path("exports/raw.csv")
    tmp_csv.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run([sys.executable, "fetch_vacancies.py", "--out_csv", str(tmp_csv)] + args_wo_output, check=True)
    subprocess.run([sys.executable, "build_job_analytics.py", "--input", str(tmp_csv), "--output", out], check=True)

    report_dir = Path("Reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_docx = report_dir / ("Отчёт_" + Path(out).stem + ".docx")
    subprocess.run(
        [
            sys.executable,
            "build_report_docx.py",
            "--input_csv",
            str(tmp_csv),
            "--output_docx",
            report_docx,
            "--query",
            query_text,
            "--city",
            city_text,
        ],
        check=True,
    )


def main() -> None:
    out, args_wo_output, query_text, city_text = parse_legacy_args(sys.argv[1:])
    run_pipeline(out, args_wo_output, query_text, city_text)


if __name__ == "__main__":
    main()
