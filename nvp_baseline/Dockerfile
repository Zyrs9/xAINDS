# =============================================================
# NVP BASELINE — Frozen Legacy NIDS (Scapy-based)
# Purpose: Benchmark containerization for CPU comparison only.
# DO NOT MODIFY THIS IMAGE. It exists solely to demonstrate
# the user-space Scapy bottleneck during live presentations.
# =============================================================
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for Scapy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev \
    tcpdump \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Start the FastAPI inference engine (Scapy sniffer runs separately)
CMD ["python", "main.py"]
