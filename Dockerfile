# syntax=docker/dockerfile:1

# Pin by digest, not tag: "3.11-slim-bookworm" is mutable and a base-image
# update should be a reviewed bump, not something that silently changes what
# ships on the next build.
ARG PYTHON_IMAGE=python@sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b

# ---- builder -----------------------------------------------------------
# Build a venv in its own stage so the final image never carries a
# toolchain, headers, or pip's package cache.
FROM ${PYTHON_IMAGE} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /build

# main.py talks to Slack, OTel and a webhook server unconditionally in this
# deployment (see k8s/agent-deployment.yaml env vars), so those extras ship
# by default. `voyage` and `dev` do not: the README default index is FTS5,
# and pytest/ruff have no business in a runtime image.
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir '.[slack,serve,otel]'

# ---- runtime -------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    APPROVALS_DB=/data/approvals.db \
    RUNS_DB=/data/runs.db \
    KNOWLEDGE_DB=/data/knowledge.db \
    TRANSCRIPTS_DB=/data/transcripts.db \
    MCP_SERVERS_FILE=/etc/agent/servers.json

# Root-owned code, non-root process: a container-breakout via the app can't
# rewrite what it runs. /data and /etc/agent exist so a docker-run smoke test
# without the k8s PVC/ConfigMap still has somewhere to write and read from;
# under k8s, the PVC mount and fsGroup govern real ownership.
RUN groupadd --system --gid 1000 agent \
    && useradd --system --uid 1000 --gid agent --no-create-home --shell /usr/sbin/nologin agent \
    && mkdir -p /data /etc/agent \
    && chown -R agent:agent /data

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY src/ src/

USER agent:agent

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/healthz', timeout=2)" || exit 1

# No shell form, no shell in the image: signals reach uvicorn directly and
# there's no injection surface via CMD.
CMD ["uvicorn", "agentic_devops.main:app", "--host", "0.0.0.0", "--port", "8080"]
