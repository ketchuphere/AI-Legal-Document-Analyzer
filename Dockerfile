# ──────────────────────────────────────────────────────────────
# AI Document Analyzer — Dockerfile
# Build:  docker build -t ai-doc-analyzer .
# Run:    docker run -p 8501:8501 --env-file .env ai-doc-analyzer
# ──────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL maintainer="AI Document Analyzer"
LABEL description="AI-powered document analysis with Claude"

# System dependencies (needed by PyMuPDF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    libfreetype6-dev \
    libharfbuzz-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home appuser
WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Ensure uploads dir exists and is owned by appuser
RUN mkdir -p /app/uploads && chown -R appuser:appuser /app
USER appuser

# Streamlit config — disable file watcher in production, bind to 0.0.0.0
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.maxUploadSize=50"]
