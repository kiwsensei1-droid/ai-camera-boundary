#!/bin/sh

echo "=== Initializing Tailscale ==="
# Check if /dev/net/tun exists
if [ ! -d /dev/net ]; then
    mkdir -p /dev/net
    mknod /dev/net/tun c 10 200 || true
fi

# Start Tailscale daemon in userspace mode with SOCKS5 on port 1055
tailscaled --tun=userspace-networking --socks5-server=localhost:1055 &
sleep 3

if [ -n "$TAILSCALE_AUTHKEY" ]; then
    echo "Authenticating Tailscale..."
    tailscale up --authkey="${TAILSCALE_AUTHKEY}" --hostname=ai-camera-cloud --accept-routes
    echo "Tailscale Connected!"
    tailscale status
fi

echo "=== Starting AI Web Application ==="
# Inject socks5 proxy into ffmpeg/opencv if available
export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"
python app.py
