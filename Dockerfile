FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV API_CONFIG_PATH=configs/api_space.yaml
ENV PORT=7860

# Add user 1000 for Hugging Face Spaces (prevent permission denied)
RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Add unzip for Reflex (needed to install bun)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=user:user pyproject.toml /app/pyproject.toml
COPY --chown=user:user rxconfig.py /app/rxconfig.py
COPY --chown=user:user configs /app/configs
COPY --chown=user:user artifacts /app/artifacts
COPY --chown=user:user api /app/api
COPY --chown=user:user inference /app/inference
COPY --chown=user:user models /app/models
COPY --chown=user:user reflex_app /app/reflex_app
COPY --chown=user:user utils /app/utils
COPY --chown=user:user assets /app/assets
COPY --chown=user:user start_hf.sh /app/start_hf.sh

# Switch to the non-root HF user
USER user

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[frontend]" \
    && chmod +x /app/start_hf.sh

EXPOSE 7860

CMD ["/app/start_hf.sh"]
