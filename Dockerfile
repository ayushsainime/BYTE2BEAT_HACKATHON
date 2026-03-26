FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV API_CONFIG_PATH=configs/api_railway.yaml
ENV PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY rxconfig.py /app/rxconfig.py
COPY configs /app/configs
COPY artifacts /app/artifacts
COPY api /app/api
COPY inference /app/inference
COPY models /app/models
COPY reflex_app /app/reflex_app
COPY utils /app/utils
COPY assets /app/assets
COPY start_railway_reflex.sh /app/start_railway_reflex.sh

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[frontend]" \
    && chmod +x /app/start_railway_reflex.sh

EXPOSE 7860

CMD ["/app/start_railway_reflex.sh"]
