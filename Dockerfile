# Use official Python runtime with OpenCV & Torch dependencies
FROM python:3.10-slim

# Install system dependencies & Tailscale
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    iptables \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    fonts-thai-tlwg \
    ca-certificates \
    && curl -fsSL https://tailscale.com/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement files & install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt ultralytics onvif-zeep

# Copy application files
COPY . .

# Expose Web Port
EXPOSE 5000

# Start script
RUN chmod +x start_cloud.sh
CMD ["./start_cloud.sh"]
