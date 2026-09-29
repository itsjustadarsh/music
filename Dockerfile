# Minimal image for the FastAPI service only. Training and data prep run
# natively on the host (see FINALSTEPS.md) -- this image just serves already
# trained checkpoints under outputs/ and the Spotify dataset CSV under data/.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY frontend ./frontend
COPY pyproject.toml .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
