FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

RUN groupadd --system datalens \
    && useradd --system --gid datalens --create-home datalens

COPY --from=builder /app/.venv /app/.venv
COPY config ./config
COPY data/manifests ./data/manifests
COPY demo ./demo
COPY src ./src
COPY .streamlit ./.streamlit

RUN mkdir -p /app/artifacts /mlartifacts \
    && chown -R datalens:datalens /app /mlartifacts

USER datalens
EXPOSE 8000

CMD ["uvicorn", "datalens.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
