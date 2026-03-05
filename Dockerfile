FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Requirements.txt .
RUN pip install --no-cache-dir -r Requirements.txt

COPY . .

RUN mkdir -p data/investments

EXPOSE 8502

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8502/_stcore/health || exit 1

CMD ["streamlit", "run", "PSAR.py", \
     "--server.port=8502", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]