FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV API_CONFIG_PATH=configs/api_space.yaml

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY configs /app/configs
COPY artifacts /app/artifacts
COPY api /app/api
COPY inference /app/inference
COPY models /app/models
COPY utils /app/utils

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

EXPOSE 7860

CMD ["python", "-m", "api.gradio_app"]

