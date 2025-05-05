Go ML Model Processor

# Overview

A distributed system in Go for processing asynchronous Machine Learning tasks (training, inference). Jobs are submitted via a REST API, queued using RabbitMQ, and processed by scalable workers. Uses PostgreSQL for metadata and MinIO/S3 for artifacts.

## Technology Stack

- Go 1.21+
- Chi v5 (HTTP Router)
- PostgreSQL (via `jackc/pgx/v5`)
- RabbitMQ (via `rabbitmq/amqp091-go`)
- MinIO / S3 (via `aws-sdk-go-v2`)
- Docker & Docker Compose

# Setup

1. Prerequisites: Install Go (1.21+), Docker, and Docker Compose (v2+).
2. Clone: `git clone <repository-url>` then `cd ner-backend`

## Setting models for downloading using MinIO 

1. Get your **Access Key** and **Secret Key** from the MinIO web UI (typically at `http://localhost:9090`).  
2. Run:  
```bash
AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin aws s3 --endpoint-url http://localhost:9000 cp s3://ner-models s3://ner-models --recursive
```

## Running with Docker Compose

### Start Head Node Services (API, DB, Queue, Storage): \
`docker compose --profile head up -d --build`

### Start Worker Services:
`docker compose --profile worker up -d --build` \
(To run N workers, use: `docker compose --profile worker up -d --scale worker=N`) \
(Note: If workers run on separate hosts, update .env URLs to point to the head node's IP/hostname).

To take services down, run: \
`docker compose --profile head down` \
`docker compose --profile worker down`

Access Services (default ports): \
API Health: http://localhost:8001/health \
RabbitMQ UI: http://localhost:15672 \
MinIO UI: http://localhost:9090

# API Endpoints

See `internal/api/docs.md` for api documentation.

# Licensing

We support 3 types of licensing
1. Keygen based license: uses keygen for verification, allows for full access.
2. Local licensing: a signed key that is verified in the backend without network calls, allows for full access.
3. Free licensing: used if no key is provided. Restricts users to 1Gb of data accross all reports.

The `cmd/licensing/main.go` cli tool provides utilities for creating an managing licensings.

### Creating Keys
```bash
go run cmd/licensing/main.go keys --output "test_key"
```

This command will generate `<output>_private_key.pem` and `<output>_public_key.pem` containing private/public keys that can be used for local licensing. Currently we use the keys saved in `/share/keys/ner_licensing` on blade. This command should not be run unless you are testing or want to change the keys. Note if keys are being changed, then the public key in `file_license.go` needs to updated as well.

### Creating a license

```bash
go run cmd/licensing/main.go create --private-key /path/to/private_key.pem --days 10
```

Creates a license using the private key at the specified path. The license will expire in the specified number of days. 

### Validating a license

```bash
go run cmd/licensing/main.go create --public-key /path/to/public_key.pem --license "your license string"
```

Validates a license and prints out information such as expiration date.