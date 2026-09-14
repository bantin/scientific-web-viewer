#!/usr/bin/env bash
# Stop the image viewer server
pkill -f "python.*app.py.*--port 8089" && echo "Server stopped." || echo "Server not running."
