#!/usr/bin/env bash
# Start the image viewer server
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source ~/miniforge3/bin/activate dendrites-py312
exec python "$SCRIPT_DIR/app.py" --port 8089 --host 127.0.0.1
