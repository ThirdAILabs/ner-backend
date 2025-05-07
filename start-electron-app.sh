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

# Check and kill any process using port 3007
echo "Checking for processes using port 3007..."
PORT_3007_PID=$(lsof -ti:3007 || true)
if [ ! -z "$PORT_3007_PID" ]; then
  echo "Killing process using port 3007 (PID: $PORT_3007_PID)..."
  kill -9 $PORT_3007_PID || true
  sleep 1
fi

# Start the Next.js development server
echo "Starting Next.js development server..."
npm run dev &
NEXT_PID=$!

# Add Next.js to the trap to kill it on exit
trap "kill $BACKEND_PID $NEXT_PID 2>/dev/null || true" EXIT

# Wait for Next.js to start
echo "Waiting for Next.js to start..."
sleep 5

# Start Electron pointing to the development server
echo "Starting Electron app..."
npm run electron:start

# Script will automatically kill the backend and Next.js when it exits thanks to the trap 