FROM python:3.11-slim

LABEL maintainer="SIH Antarctic AI Team"
LABEL description="Antarctic AI Navigation & Iceberg Trajectory Forecasting System"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/           ./src/
COPY app/           ./app/
COPY data/          ./data/
COPY models/        ./models/
COPY artifacts/     ./artifacts/

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# Run uvicorn
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
