# Shop Demand Forecasting — production-style local image
#
#   docker build -t shop-demand-forecasting .
#   docker run --rm -p 8501:8501 shop-demand-forecasting
#
# Runs as a non-root user with a read-only application tree. Nothing is written
# to disk at runtime, so no writable volume is needed.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so the layer is cached across code changes.
COPY requirements.lock.txt ./
RUN python -m pip install --no-cache-dir -r requirements.lock.txt

# Create an unprivileged user before copying application code.
RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --no-create-home --shell /usr/sbin/nologin app

COPY --chown=app:app . .
RUN rm -rf .lockenv .pytest_cache tests/__pycache__

USER app

EXPOSE 8501

# Streamlit serves a plain-text health endpoint at /_stcore/health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).read().strip() == b'ok' else 1)"

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
