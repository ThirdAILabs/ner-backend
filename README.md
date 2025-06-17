# ThirdAI NER
This project covers two products: the **Enterprise Platform**, which is a distributed system for processing NER tasks, and **PocketShield**, a desktop app that demonstrates our NER capabilities and applications.

# PocketShield

An Electron.js app that demonstrates our NER capabilities and applications.

## Setup

### Install Dependencies

First, install the necessary dependencies by running the following in the project root directory:

```bash
cd frontend/
npm install
cd ../electron-ner-app
npm install
cd ..
```

### Development Mode

1. Make sure you have the Go backend built. Run the following in the project root directory:
   ```
   go clean -cache
   go build cmd/local/main.go
   ```

2. Activate a clean python 3.11 environment. Use python venv (`pyhon -m venv <venv_name>`); there had been issues faced when using conda environments.

3. Start the integrated app (see notes about `MODEL_DIR` and `MODEL_TYPE` under the Environment Notes section):
   ```
   cd electron-ner-app
   MODEL_DIR=/path/to/models MODEL_TYPE=<model_type> npm run dev
   ```
   This will:
   - Copy the Go backend to the `bin` directory
   - Start the React development server
   - Start Electron
   - Start the Go backend

### Production Build (macOS)

1. Make sure you have the Go backend built. Run the following in the project root directory:
   ```
   go clean -cache
   go build cmd/local/main.go
   ```

2. Activate a clean python 3.11 environment. Use python venv (`pyhon -m venv <venv_name>`); there had been issues faced when using conda environments.

3. Build the integrated app (see notes about `MODEL_DIR` and `MODEL_TYPE` under the Environment Notes section):
   ```
   cd electron-ner-app
   MODEL_DIR=/path/to/models MODEL_TYPE=<model_type> npm run build
   ```
   This will:
   1. Build the Go backend for macOS
   2. Copy it to the Electron app
   3. Build the React frontend
   4. Package everything into a DMG file with the backend included

The resulting DMG file will be in the `electron-ner-app/dist` directory. 

When installed, the app will automatically find and use the backend without any additional steps required.

### Environment Notes

`MODEL_TYPE` is one of `bolt_udt`, `python_transformer`, `python_cnn`, `presidio`, `onnx_cnn`, or `regex`. We currently use `onnx_cnn` in production.

`MODEL_DIR` is a directory that contains a subdirectory with the same name as the `model_type` of choice. For example, if you want to build the app with `onnx_cnn`, then `/path/to/models` must contain a subdirectory called `onnx_cnn`.

#### Where can you obtain these model subdirectories?
- `onnx_cnn` or `python_cnn`: Download this S3 directory `s3://ner-models/cnn_model_torchscript/` and rename it to `onnx_cnn` or `python_cnn`. (It contains the required files for both models since `python_cnn`'s required files is a subset of `onnx_cnn`).
- `bolt_udt`: Download from `/share/ner/models/bolt_udt` on Blade.
- `presidio` or `regex`: These models do not require any files.
- `python_transformer: Download this S3 directory `s3://ner-models/transformer_model/` and rename it to `python_transformer`.

Note that `/path/to/models` would be the parent directory of the downloaded model directory. E.g. if you download `onnx_cnn` and it is located in `/Users/abcde/models/onnx_cnn`, then you should run the above command with `MODEL_DIR=/Users/abcde/models`


# Enterprise Platform

A distributed system in Go for processing asynchronous Machine Learning tasks (training, inference). Jobs are submitted via a REST API, queued using RabbitMQ, and processed by scalable workers. Uses PostgreSQL for metadata and MinIO/S3 for artifacts.

### Technology Stack

- Go 1.21+
- Chi v5 (HTTP Router)
- PostgreSQL (via `jackc/pgx/v5`)
- RabbitMQ (via `rabbitmq/amqp091-go`)
- MinIO / S3 (via `aws-sdk-go-v2`)
- Docker & Docker Compose

## Setup

1. Prerequisites: Install Go (1.21+), Docker, and Docker Compose (v2+).
2. Clone: `git clone <repository-url>` then `cd ner-backend`



### Running with Docker Compose

#### Start Head Node Services (API, DB, Queue, Storage): \
`docker compose --profile head up -d --build`

#### Start Worker Services:
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

### Setting models for downloading using MinIO 

You need to update the mount volume path in docker compose file.

## API Endpoints

See `internal/api/docs.md` for api documentation.

## Licensing

We support 3 types of licensing
1. Keygen based license: uses keygen for verification, allows for full access.
2. Local licensing: a signed key that is verified in the backend without network calls, allows for full access.
3. Free licensing: used if no key is provided. Restricts users to 1Gb of data accross all reports.

The license key is determined by the `LICENSE_KEY` env when the worker starts. If the key starts with `local:` it will be treated as a local license key. Otherwise it will be treated as a keygen license. Keygen licenses need to have the FULL_ACCESS entitlement. If no license key is specified then it will use the free license. The free license simply restricts the total size of all files accross all jobs by checking the DB. The current max size is 1GB.

The `cmd/licensing/main.go` cli tool provides utilities for creating an managing licensings.

#### Creating Keys
```bash
go run cmd/licensing/main.go keys --output "test_key"
```

This command will generate `<output>_private_key.pem` and `<output>_public_key.pem` containing private/public keys that can be used for local licensing. Currently we use the keys saved in `/share/keys/ner_licensing` on blade. This command should not be run unless you are testing or want to change the keys. Note if keys are being changed, then the public key in `file_license.go` needs to updated as well.

#### Creating a license

```bash
go run cmd/licensing/main.go create --private-key /path/to/private_key.pem --days 10
```

Creates a license using the private key at the specified path. The license will expire in the specified number of days. 

#### Validating a license

```bash
go run cmd/licensing/main.go create --public-key /path/to/public_key.pem --license "your license string"
```

Validates a license and prints out information such as expiration date.