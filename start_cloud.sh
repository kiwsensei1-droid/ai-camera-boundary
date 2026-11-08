#!/bin/sh

# Start Tailscale in userspace mode
tailscaled --tun=userspace-networking --socks5-server=localhost:1055 &
sleep 2

# Connect to Tailscale network using Auth Key
if [ -n "$TAILSCALE_AUTHKEY" ]; then
    echo "Connecting to Tailscale..."
    tailscale up --authkey=${TAILSCALE_AUTHKEY} --hostname=ai-camera-cloud --accept-routes
    echo "Tailscale Connected!"
else
    echo "Warning: TAILSCALE_AUTHKEY is not set."
fi

# Run AI Web Application
python app.py
