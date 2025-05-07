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

# Start the Next.js development server
echo "Starting Next.js development server..."
npm run dev &
NEXT_PID=$!

# Add Next.js to the trap to kill it on exit
trap "kill $BACKEND_PID $NEXT_PID" EXIT

# Wait for Next.js to start
echo "Waiting for Next.js to start..."
sleep 5

# Start Electron pointing to the development server
echo "Starting Electron app..."
npm run electron:start

# Script will automatically kill the backend and Next.js when it exits thanks to the trap 