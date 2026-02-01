# Multi-stage production Dockerfile for Photo-Clean-Up
# Optimized for self-hosted deployments

FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /app/data /photos && \
    chown -R appuser:appuser /app /photos

WORKDIR /app

# Copy requirements and install Python dependencies
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser clean.py tools.py ./

# Switch to non-root user
USER appuser

# Create directories for persistent data
RUN mkdir -p /app/data/thumbnails

# Expose port (configurable via PORT env var, defaults to 8080)
EXPOSE 8080

# Health check to ensure the app is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080').read()" || exit 1

# Environment variables with sensible defaults
ENV ROOT_DIR=/photos \
    PORT=8080 \
    PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "clean.py"]
