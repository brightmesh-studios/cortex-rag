FROM python:3.14-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Fixed uid so a host-mounted /state can be chowned to a known owner.
RUN groupadd --system --gid 10001 cortex \
    && useradd --system --uid 10001 --gid cortex cortex

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y

COPY --chown=cortex:cortex src/ ./src/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

USER cortex

CMD ["python", "src/server.py"]
