# --- Build Stage ---
FROM ubuntu:24.04 as builder

ARG GOLANG_DOWNLOAD_URL=https://go.dev/dl/go1.24.2.linux-arm64.tar.gz
ARG GOLANG_DOWNLOAD_SHA256=756274ea4b68fa5535eb9fe2559889287d725a8da63c6aae4d5f23778c229f4b

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "${GOLANG_DOWNLOAD_URL}" -o golang.tar.gz \
    && echo "${GOLANG_DOWNLOAD_SHA256} golang.tar.gz" | sha256sum -c - \
    && tar -C /usr/local -xzf golang.tar.gz \
    && rm golang.tar.gz

ENV PATH="/usr/local/go/bin:${PATH}"

WORKDIR /app

# Copy module files and download dependencies first for caching
COPY go.mod go.sum ./
RUN go mod download

# Copy the rest of the application source code
COPY . .

# Build the API server binary
RUN CGO_ENABLED=1 GOOS=linux go build -v -o /app/api ./cmd/api

# Build the Worker binary
RUN CGO_ENABLED=1 GOOS=linux go build -v -o /app/worker ./cmd/worker

WORKDIR /app

COPY ./entrypoint.sh /entrypoint.sh

# Ensure entrypoint is executable
RUN chmod +x /entrypoint.sh

# Expose API port
EXPOSE 8001

# Set the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]

# Default command (starts as worker)
CMD ["--worker"]