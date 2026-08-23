FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y

COPY src/ ./src/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

CMD ["python", "src/server.py"]
