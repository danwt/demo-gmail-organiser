FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY main.py taxonomy.yaml ./
RUN uv sync --frozen --no-dev
ENTRYPOINT ["uv", "run", "python", "main.py"]
