#!/bin/sh

# Start Tailscale background daemon
tailscaled --tun=userspace-networking &
sleep 2

# Authenticate Tailscale if key is provided
if [ -n "$TAILSCALE_AUTHKEY" ]; then
    echo "Connecting Tailscale network..."
    tailscale up --authkey="${TAILSCALE_AUTHKEY}" --hostname=ai-camera-cloud --accept-routes
    echo "Tailscale Connected Successfully!"
fi

# Run Web App directly
echo "Starting AI Web Application..."
python app.py
