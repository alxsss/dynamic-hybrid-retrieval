FROM python:3.13-slim AS runtime
RUN pip install --no-cache-dir uv==0.11.23
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
RUN useradd --create-home --uid 10001 dhr && mkdir -p /data && chown dhr:dhr /data
USER dhr
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["dhr", "serve", "--repository", "/data/repository", "--event-log", "/data/wal", "--segmented-wal", "--host", "0.0.0.0", "--port", "8000"]
