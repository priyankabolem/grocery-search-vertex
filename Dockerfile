FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY grocery_search/ grocery_search/

RUN useradd --create-home appuser
USER appuser

CMD exec uvicorn grocery_search.api:app --host 0.0.0.0 --port ${PORT}
