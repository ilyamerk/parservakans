# Parser Vakans — HH-only (FastAPI)

Проект парсит вакансии **только с hh.ru** и формирует:
- raw CSV,
- Excel-аналитику,
- DOCX-отчёт.

## Структура

- `fetch_vacancies.py` — HH-парсер и нормализация данных.
- `build_job_analytics.py` — расчёты и выгрузка в Excel.
- `build_report_docx.py` — DOCX-отчёт.
- `main.py` — CLI-обёртка полного пайплайна.
- `app/main.py` — FastAPI веб/API-слой.
- `core/pipeline_service.py` — запуск пайплайна из веб-слоя.

## Локальный запуск

```bash
pip install -r requirements.txt
python main.py --query "Бариста" --city "Москва"
```

## Веб-режим

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Открыть: `http://localhost:8000`

## API

- `GET /` — веб-страница запуска.
- `POST /run` — запуск из HTML-формы.
- `POST /api/run` — запуск через JSON API.
- `GET /api/result/{run_id}` — результат и preview.
- `GET /download/{run_id}/csv|xlsx|docx` — скачивание файлов.
- `GET /health` — health-check.
