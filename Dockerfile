FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local

# Copy ONLY production files
COPY main.py .
COPY class_names.json .
COPY trained_model.h5 .
COPY DataBase/ DataBase/
COPY RAG/ RAG/
COPY vector_store/ vector_store/

ENV PATH=/root/.local/bin:$PATH

RUN printf '#!/bin/sh\nexec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8080}"\n' > /start.sh && \
    chmod +x /start.sh

CMD ["/bin/sh", "/start.sh"]