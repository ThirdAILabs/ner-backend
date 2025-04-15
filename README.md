Go ML Model Processor

Overview

A distributed system in Go for processing asynchronous Machine Learning tasks (training, inference). Jobs are submitted via a REST API, queued using RabbitMQ, and processed by scalable workers. Uses PostgreSQL for metadata and MinIO/S3 for artifacts.

Technology Stack

- Go 1.21+
- Chi v5 (HTTP Router)
- PostgreSQL (via `jackc/pgx/v5`)
- RabbitMQ (via `rabbitmq/amqp091-go`)
- MinIO / S3 (via `aws-sdk-go-v2`)
- Docker & Docker Compose

Project Structure

The project contains several main directories:

- cmd: Contains the executable entrypoints for the api server and the worker process.
- internal: Holds private application logic including database interaction (database/), queue handling (messaging/), S3/MinIO utilities (s3/), and core processing logic (core/).
- pkg: Contains shared code and data types, primarily the data models (models/).

Key files at the root include: .env for environment variables, .gitignore, docker-compose.yml for service definitions, Dockerfile for building the application image, and entrypoint.sh used inside the container.

Setup

1. Prerequisites: Install Go (1.21+), Docker, and Docker Compose (v2+).
2. Clone: `git clone <repository-url>` then `cd model_processor_go`
3. Configure: Create an `.env` file (you can copy the example) and fill in your specific database, RabbitMQ, and MinIO/S3 credentials, endpoints, and bucket names.

Build

Build the Docker image using this command in the project root:
`docker compose build`

Running with Docker Compose

Start Head Node Services (API, DB, Queue, Storage):
`docker compose --profile head up -d`

Start Worker Services:
`docker compose --profile worker up -d`
(To run N workers, use: `docker compose --profile worker up -d --scale worker=N`)
(Note: If workers run on separate hosts, update .env URLs to point to the head node's IP/hostname).

Access Services (default ports):
API Health: http://localhost:8000/health
RabbitMQ UI: http://localhost:15672
MinIO UI: http://localhost:9090

Stop Services:
`docker compose down`
(To also remove data volumes, use: `docker compose down -v`)

Adding a New Task Type

1. Payload: Define the new task's data structure (struct) in `pkg/models/models.go`.
2. Publisher: Define a queue name constant. Add a `Publish<NewTask>Task` function in `internal/messaging/publisher.go` to send the payload.
3. Worker Handler: Add a `handle<NewTask>Task` function in `internal/messaging/worker.go` with the logic to process the task.
4. Worker Router: Add a case for the new queue name in the `processMessage` switch statement (in `internal/messaging/worker.go`) to call your new handler.
5. Queue Config: Ensure the new queue is declared by the worker/publisher and add the queue name to the `QUEUE_NAMES` environment variable (in `.env`) so workers consume from it.
6. API: Add an API handler function (`cmd/api/handlers.go`) and register a route (`cmd/api/main.go`) for submitting the new task type.

API Endpoints

POST /models : Submit training job.
GET /models/{model_id} : Get training job status.
POST /inference : Submit inference job.
GET /inference/{job_id} : Get inference job status.
GET /health : Health check.

ML Integration Note

The ML functions in `internal/core/ml.go` are placeholders. Integrating real ML models requires extra work (e.g., using cgo with C++ libraries, ONNX Runtime, or calling external services).