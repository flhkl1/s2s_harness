# harness-api — no GPU required
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopus0 libopus-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY harness/ ./harness/

EXPOSE 8000
CMD ["uvicorn", "harness.server:app", "--host", "0.0.0.0", "--port", "8000"]
