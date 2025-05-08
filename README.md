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

The license key is determined by the `LICENSE_KEY` env when the worker starts. If the key starts with `local:` it will be treated as a local license key. Otherwise it will be treated as a keygen license. Keygen licenses need to have the FULL_ACCESS entitlement. If no license key is specified then it will use the free license. The free license simply restricts the total size of all files accross all jobs by checking the DB. The current max size is 1GB.

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

# NER Electron App

This is an Electron application for Named Entity Recognition that integrates a Go backend service.

## Project Structure

```
electron-ner-app/               # Frontend Electron application
├── bin/                        # Directory for storing the backend executable
│   └── main                    # Go backend executable
├── build/                      # Build resources for electron-builder
├── main.js                     # Electron main process
├── preload.js                  # Electron preload script
├── package.json                # Project configuration
├── scripts/                    # Utility scripts
│   ├── after-pack.js           # Hook for electron-builder
│   ├── build-dmg.js            # Script to build DMG
│   ├── copy-backend.js         # Script to copy backend
│   ├── debug-packaging.js      # Debug utilities
│   └── start-backend.js        # Script to start backend
└── src/                        # React application source code
    ├── components/             # React components
    ├── lib/                    # Utility libraries
    │   └── api.js              # Backend API communication
    └── pages/                  # Application pages

../main                         # Go backend executable in parent directory
```

## Setup and Development

### Prerequisites

- Node.js (v14+)
- npm (v6+)
- Go (v1.16+)
- macOS for building DMG packages (Windows/Linux supported for their respective formats)

### Installation

1. Build the Go backend first:
   ```bash
   # From the parent directory (ner-backend)
   go build -o main
   ```

2. Install the Electron app dependencies:
   ```bash
   cd electron-ner-app
   npm install
   ```

### Development Mode

To run the app in development mode:

```bash
npm run dev
```

This will:
- Copy the Go backend to the `bin` directory
- Start the Vite development server
- Start Electron
- Start the Go backend

## Building for Production

### macOS DMG

To build a macOS DMG with the integrated backend:

```bash
npm run build-dmg
```

This will:
1. Build the Go backend for macOS
2. Copy it to the Electron app
3. Build the React frontend
4. Package everything into a DMG file with the backend included

The resulting DMG file will be in the `dist` directory. 

When installed, the app will automatically find and use the backend without any additional steps required.

### Other Platforms

For Windows:
```bash
npm run build-win
```

For Linux:
```bash
npm run build-linux
```

## Technical Implementation Details

### Backend Integration

The Go backend is integrated with Electron through several components:

1. **Backend Management**:
   - `start-backend.js`: Locates and starts the backend
   - `copy-backend.js`: Copies the backend during build
   - `after-pack.js`: Ensures backend is properly packaged

2. **Communication**:
   - Frontend communicates with backend over HTTP (port 8000)
   - API utilities in `src/lib/api.js` handle requests
   - Backend URL exposed through preload script

3. **Packaging**:
   - Backend binary is included in the app's Resources directory
   - No separate installation needed

### Key Configuration

Important configuration options in `package.json`:

```json
"build": {
  "files": [...],
  "extraResources": [
    {
      "from": "bin",
      "to": "bin"
    }
  ],
  "afterPack": "./scripts/after-pack.js"
}
```

## Troubleshooting

If you encounter issues:

1. **Backend Connection Issues**:
   - Check the app logs via developer tools console (View > Toggle Developer Tools)
   - Ensure no other process is using port 8000
   - Verify backend process is running

2. **Build Problems**:
   - Ensure Go backend was built first
   - Check for any error messages during build
   - Verify all dependencies are installed

3. **Advanced Debugging**:
   ```bash
   # Inspect the packaged app
   npm run debug-pkg inspect
   
   # Test running the backend
   npm run debug-pkg run-backend
   
   # Run with debug logs
   npm run debug-pkg run-app
   ```

## License

ISC