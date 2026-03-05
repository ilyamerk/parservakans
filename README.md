# Parser Vakans — web version (FastAPI)

Проект сохранён как Python-парсер вакансий и дополнен веб-интерфейсом.

## Структура

- `app/main.py` — FastAPI-приложение и веб/API-эндпоинты.
- `core/pipeline_service.py` — сервис запуска текущего пайплайна через существующие скрипты.
- `core/schemas.py` — структуры параметров/результата запуска.
- `templates/index.html` — веб-форма и вывод результатов.
- `static/style.css` — минимальные стили.
- `.devcontainer/devcontainer.json` — запуск в GitHub Codespaces.
- `tests/test_web_app.py` — тест веб-слоя.

## Локальный запуск

```bash
pip install -r requirements.txt
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

## GitHub Codespaces

1. Открыть репозиторий в Codespaces.
2. Дождаться выполнения `postCreateCommand` (установка зависимостей).
3. Запустить:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
4. Открыть автоматически проброшенный порт `8000`.

