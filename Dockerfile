FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV API_CONFIG_PATH=configs/api.yaml

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY configs /app/configs
COPY api /app/api
COPY datasets /app/datasets
COPY evaluation /app/evaluation
COPY inference /app/inference
COPY models /app/models
COPY training /app/training
COPY utils /app/utils
COPY tests /app/tests
COPY data /app/data

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
