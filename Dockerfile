FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

RUN pip install --no-cache-dir fastapi uvicorn pydantic pytest

COPY axiom-runtime/src/ /app/src/
COPY axiom-runtime/server.py /app/server.py

EXPOSE 8001

CMD ["python", "server.py"]
