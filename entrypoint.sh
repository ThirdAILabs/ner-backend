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
  echo "--- Starting Go API Server ---"
  # Execute the compiled Go binary for the API
  # Use "$@" to pass any remaining arguments if needed
  exec /app/api "$@"
else
  echo "--- Starting Go Worker Process ---"
  # Execute the compiled Go binary for the worker
  # Use "$@" to pass any remaining arguments if needed
  exec /app/worker "$@"
fi