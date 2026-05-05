"""Загрузка API-ключей для парсера.

Ключи берутся из файла ``api_keys.json`` в корне проекта. Если файла нет
или какого-то ключа в нём не хватает, значение подхватывается из переменной
окружения с тем же именем. Это позволяет хранить ключи в одном понятном
файле, не теряя совместимости с ``.env``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
KEYS_PATH = ROOT / "api_keys.json"

KNOWN_KEYS = ("HH_CLIENT_ID", "HH_CLIENT_SECRET", "HH_APP_TOKEN")

_cache: Dict[str, str] | None = None


def _load_file() -> Dict[str, str]:
    if not KEYS_PATH.exists():
        return {}
    try:
        data = json.loads(KEYS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[api_keys] не удалось прочитать {KEYS_PATH.name}: {exc}")
        return {}
    if not isinstance(data, dict):
        print(f"[api_keys] {KEYS_PATH.name} должен быть JSON-объектом")
        return {}
    return {str(k): str(v) for k, v in data.items() if v is not None}


def load_api_keys() -> Dict[str, str]:
    """Возвращает словарь с ключами, кешируя результат."""
    global _cache
    if _cache is not None:
        return _cache
    file_keys = _load_file()
    merged: Dict[str, str] = {}
    for name in KNOWN_KEYS:
        value = file_keys.get(name, "").strip()
        placeholder = value.startswith("PASTE_") and value.endswith("_HERE")
        if not value or placeholder:
            value = os.getenv(name, "").strip()
        merged[name] = value
    _cache = merged
    return merged


def get_key(name: str) -> str:
    return load_api_keys().get(name, "")
