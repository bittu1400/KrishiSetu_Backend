FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY --from=builder /root/.local /root/.local

# Copy application files
COPY main.py .
COPY DataBase/ DataBase/
COPY RAG/ RAG/
COPY class_names.json .
COPY trained_model.h5 .

ENV PATH=/root/.local/bin:$PATH

EXPOSE $PORT

# Use shell form to allow variable expansion
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
