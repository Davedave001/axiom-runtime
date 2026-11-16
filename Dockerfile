FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY pyproject.toml .
RUN pip install --no-cache-dir fastapi uvicorn pydantic pytest

COPY src/ /app/src/
COPY tests/ /app/tests/
COPY server.py /app/server.py

EXPOSE 8001

CMD ["python", "server.py"]
