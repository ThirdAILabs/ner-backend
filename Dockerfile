# --- Frontend Build Stage ---
FROM node:22-alpine AS frontend-builder
WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm install --force

COPY frontend .

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


FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip libgomp1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:${PATH}"
RUN python3 -m venv /opt/venv && \
    pip install --upgrade pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY plugin/plugin-python/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Downloading the model here to avoid grpc plugin emitting download logs
RUN python -m spacy download en_core_web_lg

# Copy only the necessary artifacts from the builder stage
COPY --from=backend-builder /app/api /app/api
COPY --from=backend-builder /app/worker /app/worker
COPY --from=backend-builder /app/entrypoint.sh /app/entrypoint.sh

COPY --from=backend-builder /app/plugin/plugin-python /app/plugin/plugin-python

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

RUN chmod +x /app/entrypoint.sh

EXPOSE 8001

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--worker"]