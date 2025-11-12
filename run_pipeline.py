#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from avito_pipeline import (
    DEFAULT_OUTPUT_PATH,
    AvitoPipelineConfig,
    build_analytics,
    export_avito_to_excel,
    fetch_avito_vacancies,
    load_config,
)


def _parse_avito_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Job analytics pipeline")
    parser.add_argument("--source", required=True, choices=["avito"], help="Источник данных")
    parser.add_argument("--config", required=True, help="Путь к config.json")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Путь для сохранения Excel (по умолчанию Exports/vacancies_avito.xlsx)",
    )
    parser.add_argument("--log-level", default="INFO", help="Уровень логирования")
    return parser.parse_args(argv)


def _run_avito_pipeline(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    config: AvitoPipelineConfig = load_config(config_path)
    logging.info(
        "Avito pipeline: queries=%s city=%s region=%s pages_limit=%s",
        config.queries,
        config.city,
        config.region,
        config.pages_limit,
    )

    data_df, logs_df = fetch_avito_vacancies(config)
    analytics = build_analytics(data_df)
    output_path = Path(args.output)
    export_avito_to_excel(data_df, analytics, logs_df, output_path)

    logging.info("Excel saved to %s", output_path)
    logging.info("Vacancies collected: %s", len(data_df))


def parse_legacy_args(argv):
    if "--output" not in argv:
        print(
            "Usage: run_pipeline.py --query ... --area ... [--city CITY] --pages ... --per_page ... --output OUT.xlsx",
        )
        sys.exit(1)
    args = argv[:]  # копия
    out_idx = args.index("--output")
    out_val = args[out_idx + 1]
    args_wo_output = args[:out_idx] + args[out_idx + 2:]

    def get_opt(name, default=""):
        if name in args_wo_output:
            i = args_wo_output.index(name)
            return args_wo_output[i + 1] if i + 1 < len(args_wo_output) else default
        return default

    q = get_opt("--query", "")
    city = get_opt("--city", "")
    return out_val, args_wo_output, q, city


def run_legacy_pipeline(out: str, args_wo_output: list[str], query_text: str, city_text: str) -> None:
    tmp_csv = Path("exports/raw.csv")
    tmp_csv.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [sys.executable, "fetch_vacancies.py", "--out_csv", str(tmp_csv)] + args_wo_output,
        check=True,
    )
    print(f"OK: {tmp_csv}")

    subprocess.run(
        [sys.executable, "build_job_analytics.py", "--input", str(tmp_csv), "--output", out],
        check=True,
    )
    print(f"OK: {out}")

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
    print(f"OK: {report_docx}")


def main() -> None:
    argv = sys.argv[1:]
    if "--source" in argv:
        args = _parse_avito_args(argv)
        if args.source == "avito":
            _run_avito_pipeline(args)
        else:  # pragma: no cover - defensive
            raise SystemExit(f"Unsupported source: {args.source}")
    else:
        out, args_wo_output, query_text, city_text = parse_legacy_args(argv)
        run_legacy_pipeline(out, args_wo_output, query_text, city_text)


if __name__ == "__main__":
    main()
