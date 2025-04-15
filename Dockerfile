# --- Build Stage ---
    FROM golang:1.21 as builder

    WORKDIR /app
    
    # Copy module files and download dependencies first for caching
    COPY go.mod go.sum ./
    RUN go mod download
    
    # Copy the rest of the application source code
    COPY . .
    
    # Build the API server binary
    RUN CGO_ENABLED=0 GOOS=linux go build -v -o /app/bin/api ./cmd/api
    
    # Build the Worker binary
    RUN CGO_ENABLED=0 GOOS=linux go build -v -o /app/bin/worker ./cmd/worker
    
    # --- Final Stage ---
    # Use a minimal base image
    FROM alpine:latest
    
    # Install any runtime dependencies needed (e.g., CA certificates)
    RUN apk --no-cache add ca-certificates tzdata
    
    WORKDIR /app
    
    # Copy the compiled binaries and entrypoint script from the builder stage
    COPY --from=builder /app/bin/api /app/api
    COPY --from=builder /app/bin/worker /app/worker
    COPY ./entrypoint.sh /entrypoint.sh
    
    # Ensure entrypoint is executable
    RUN chmod +x /entrypoint.sh
    
    # Expose API port
    EXPOSE 8000
    
    # Set the entrypoint script
    ENTRYPOINT ["/entrypoint.sh"]
    
    # Default command (starts as worker)
    CMD ["--worker"]