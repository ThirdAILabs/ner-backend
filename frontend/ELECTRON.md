# Electron App for NER Frontend

This guide explains how to run and build the Electron version of the NER (Named Entity Recognition) frontend.

## Prerequisites

- Node.js and npm installed
- The backend server running on port 8001 (default)

## Development

To run the application in development mode:

```bash
# Install dependencies
npm install --legacy-peer-deps

# Run the Electron app in dev mode
npm run electron:dev
```

This will start both the Next.js development server and the Electron app, allowing you to make changes to the code and see them immediately.

## Building for Production

To build the application for production:

```bash
# Build the Next.js app and package with Electron
npm run electron:build
```

This will create a production build in the `dist` directory.

## How Dynamic Routes Are Handled

The original Next.js application uses dynamic routes like `/token-classification/[deploymentId]`. In the Electron version:

1. We use `generateStaticParams` to pre-render the pages with specific deployment IDs at build time
2. The Electron main process communicates with the backend to fetch actual deployment IDs
3. The frontend uses a special utility (`electronRouter.tsx`) to manage these dynamic routes

## Architecture

- **Main Process**: `electron/main.js` - Responsible for creating the app window and handling IPC
- **Preload Script**: `electron/preload.js` - Exposes safe APIs to the renderer process
- **Renderer Process**: The built Next.js application
- **API Client**: `lib/electronApiClient.ts` - Special API client for Electron environment

## Backend Integration

In production mode, the Electron app connects to a locally running backend server at `http://localhost:8001`. Make sure this server is running before starting the Electron app.

## Known Limitations

- If new deployments are added to the backend, the Electron app needs to be rebuilt to include these in the static export
- Some features that rely on server-side rendering might not work as expected
- API routes are not available in the static export 