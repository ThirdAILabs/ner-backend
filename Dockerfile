# --- Build Stage ---
# Consider specifying the platform explicitly if your build machine isn't ARM64
# FROM --platform=linux/arm64 golang:1.24 as builder
FROM golang:1.24 as builder

WORKDIR /app

RUN apt-get update -y && apt-get install -y gcc-11 g++-11 cmake zlib1g-dev python3-dev
ENV CC=gcc-11 CXX=g++-11


# Copy module files and download dependencies first for caching
COPY go.mod go.sum ./
RUN go mod download

# Copy the rest of the application source code
COPY . .

# Build the API server binary for Linux ARM64
RUN CGO_ENABLED=1 GOOS=linux go build -v -o /app/api ./cmd/api 

# Build the Worker binary for Linux ARM64
RUN CGO_ENABLED=1 GOOS=linux go build -v -o /app/worker ./cmd/worker 

# ... rest of your Dockerfile