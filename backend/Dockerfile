FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ML models to avoid runtime HuggingFace rate limits
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)"

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run Uvicorn (Cloud Run uses the PORT environment variable, defaults to 8080)
ENV HF_HUB_OFFLINE=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
