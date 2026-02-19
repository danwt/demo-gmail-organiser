FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ARG GOGCLI_VERSION=0.11.0
ARG TARGETARCH
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl tar \
    && rm -rf /var/lib/apt/lists/*
RUN case "$TARGETARCH" in \
        amd64) ARCH="amd64" ;; \
        arm64) ARCH="arm64" ;; \
        *) echo "Unsupported TARGETARCH: $TARGETARCH" && exit 1 ;; \
    esac \
    && curl -fsSL "https://github.com/steipete/gogcli/releases/download/v${GOGCLI_VERSION}/gogcli_${GOGCLI_VERSION}_linux_${ARCH}.tar.gz" -o /tmp/gogcli.tar.gz \
    && tar -xzf /tmp/gogcli.tar.gz -C /tmp \
    && install -m 0755 /tmp/gog /usr/local/bin/gog \
    && rm -f /tmp/gog /tmp/gogcli.tar.gz
COPY pyproject.toml uv.lock ./
COPY main.py taxonomy.yaml ./
RUN uv sync --frozen --no-dev
ENTRYPOINT ["uv", "run", "python", "main.py"]
