#!/bin/bash

# Exit on error
set -e

# Build the backend
echo "Building Go backend..."
go build ./cmd/local/main.go

# Start the backend in the background
echo "Starting backend server on port 8000..."
./main &
BACKEND_PID=$!

# Trap to ensure we kill the backend when the script exits
trap "kill $BACKEND_PID" EXIT

# Give the backend a moment to start
sleep 2

# Change to frontend directory
cd frontend

# Start the Electron app
echo "Starting Electron app..."
npm run electron:dev

# Script will automatically kill the backend when it exits thanks to the trap 