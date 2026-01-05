#!/bin/bash
# Startup script for the visualizer backend server

cd "$(dirname "$0")"

# Check if evolution_output directory exists
if [ ! -d "../evolution_output" ]; then
    echo "Warning: evolution_output directory not found. Using default path."
    EVOLUTION_DIR="../evolution_output"
else
    EVOLUTION_DIR="../evolution_output"
fi

# Start the server
python backend/server.py \
    --evolution-dir "$EVOLUTION_DIR" \
    --host 127.0.0.1 \
    --port 5000 \
    "$@"

