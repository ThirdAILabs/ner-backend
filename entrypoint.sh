#!/bin/sh
set -e # Exit immediately if a command exits with a non-zero status.

# Default role
ROLE="worker"

# Check first argument for role
if [ "$1" = "--head" ]; then
  ROLE="head"
  shift # Remove --head from arguments
elif [ "$1" = "--worker" ]; then
  ROLE="worker"
  shift # Remove --worker from arguments
fi

echo "--- Starting container in $ROLE mode ---"

# No DB migrations handled here in Go version (assume handled externally or by app)

if [ "$ROLE" = "head" ]; then
  # --- Head Mode: Start Next.js + Go API ---
  echo "Starting static file server (background)..."
  serve out -p 3000 &
  SERVE_PID=$!
  echo "Serve PID: $SERVE_PID"

  echo "Starting Go API Server (foreground)..."
  exec /app/api "$@" # Replace shell with Go API

else
  # --- Worker Mode: Start Go Worker Only ---
  echo "Starting Go Worker Process (foreground)..."
  exec /app/worker "$@" # Replace shell with Go Worker
fi