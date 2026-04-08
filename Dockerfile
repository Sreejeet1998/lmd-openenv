# Use openenv-base as the starting point
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Install git for dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Copy the entire repo
COPY . /app

# Initialize the environment with uv
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv; \
    fi

# Sync dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable

# Final stage
FROM ${BASE_IMAGE}
WORKDIR /app

# Copy the venv and app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

# Set environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
