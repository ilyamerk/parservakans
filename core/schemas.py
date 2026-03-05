from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParserRunRequest:
    query: str = "Бариста"
    area: int = 1
    city: str = "Москва"
    pages: int = 3
    per_page: int = 50
    pause: float = 0.9
    search_in: str = "name"
    workers: int = 8
    timeout: float = 8.0
    role: str = ""
    no_filter: bool = True
    avito_headful: bool = False
    avito_state: str = "avito_state.json"


@dataclass
class ParserRunResult:
    run_id: str
    csv_path: Path
    xlsx_path: Path
    docx_path: Path
    row_count: int
