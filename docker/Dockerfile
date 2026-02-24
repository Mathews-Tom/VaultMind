FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install vaultmind
COPY pyproject.toml README.md ./
COPY src/ src/
RUN uv pip install --system ".[all]"

# Create data directories
RUN mkdir -p /app/data /app/vault /app/config

# Copy default config
COPY config/default.toml config/default.toml

EXPOSE 8765

ENTRYPOINT ["vaultmind"]
CMD ["bot"]
