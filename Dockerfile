# Reproducible environment for the pipelines, notebooks and MLflow UI (CPU).
#   docker compose up lab mlflow        # JupyterLab on :8888, MLflow UI on :5000
#   docker compose run --rm pipeline    # kedro run (full reference run)
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    DO_NOT_TRACK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MLFLOW_TRACKING_URI=sqlite:////workspace/mlflow-container.db

# libgomp1: OpenMP runtime for LightGBM; git: notebooks clone/pull; nodejs/npm: Marp slides
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 git nodejs npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Dependencies first (cached layer), CPU-only torch.
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && uv pip install -e ".[dev]"

COPY . .

EXPOSE 8888 5000
CMD ["kedro", "run", "--env", "smoke"]
