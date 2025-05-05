# --- Frontend Build Stage ---
FROM ubuntu:24.04 as frontend-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /frontend

COPY ../frontend/package*.json ./
RUN npm install --force

COPY ../frontend .

RUN npm run build


# --- Backend Build Stage ---
FROM ubuntu:24.04 as backend-builder


ARG GOLANG_VERSION=1.24.2
ARG TARGETARCH
ARG GOLANG_DOWNLOAD_URL="https://go.dev/dl/go${GOLANG_VERSION}.linux-${TARGETARCH}.tar.gz"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN echo "Downloading Go ${GOLANG_VERSION} for ${TARGETARCH} from ${GOLANG_DOWNLOAD_URL}" \
    && curl -fsSL "${GOLANG_DOWNLOAD_URL}" -o golang.tar.gz \
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
RUN CGO_ENABLED=1 GOOS=linux GOARCH=${TARGETARCH} go build -v -o /app/api ./cmd/api

# Build the Worker binary
RUN CGO_ENABLED=1 GOOS=linux GOARCH=${TARGETARCH} go build -v -o /app/worker ./cmd/worker

# Copy and make entrypoint script executable in the builder stage
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# --- Final Stage ---
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only the necessary artifacts from the builder stage
COPY --from=backend-builder /app/api /app/api
COPY --from=backend-builder /app/worker /app/worker
COPY --from=backend-builder /entrypoint.sh /entrypoint.sh

# Copy necessary files for running 'next start'
COPY --from=frontend-builder /frontend/package*.json ./
COPY --from=frontend-builder /frontend/next.config.js ./
# Copy the build output directory
COPY --from=frontend-builder /frontend/.next ./.next

# Install only production dependencies for Next.js runtime
RUN npm install --production --ignore-scripts --prefer-offline --force

# Set Node environment to production
ENV NODE_ENV=production

EXPOSE 3000
EXPOSE 8001

ENTRYPOINT ["/entrypoint.sh"]

CMD ["--worker"]