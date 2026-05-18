# ─────────────────────────────────────────────────────────────────────────────
# Production Dockerfile — Real Estate API
# Target platform: linux/amd64 (AWS EC2 t3.large)
# ─────────────────────────────────────────────────────────────────────────────

# Use the official slim Python image.
# "slim" removes documentation, test suites, and unused locales (~120 MB saved).
# Pin the exact version for reproducible builds.
FROM python:3.11-slim

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL maintainer="your-email@example.com"
LABEL description="Real Estate Search & Price Prediction API"
LABEL version="1.0.0"

# ── System dependencies ───────────────────────────────────────────────────────
# build-essential is needed to compile numpy / scikit-learn C extensions
# if pre-built wheels are unavailable for linux/arm64.
# We clean up the apt cache in the same layer to keep the image lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (security best practice) ───────────────────────────────────
RUN groupadd --gid 1001 appgroup \
    && useradd  --uid 1001 --gid appgroup --no-create-home appuser

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy only requirements first so Docker can cache this layer.
# Rebuilds only happen when requirements.txt changes.
COPY requirements.txt .

RUN pip install --upgrade pip --no-cache-dir \
    && pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY app.py kdtree.py structures.py ./

# Transfer ownership to the non-root user
RUN chown -R appuser:appgroup /app

# ── Runtime ───────────────────────────────────────────────────────────────────
USER appuser

EXPOSE 8080

# Health-check: Docker / Oracle Container Engine will mark the container
# unhealthy if /health returns non-2xx for 3 consecutive 30s intervals.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Uvicorn flags:
#   --host 0.0.0.0      → listen on all interfaces (required inside Docker)
#   --workers 1         → single worker; the KD-Tree lives in one process
#                         (multi-worker would require shared memory or Redis)
#   --access-log        → log every request to stdout (visible via docker logs)
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--access-log"]
