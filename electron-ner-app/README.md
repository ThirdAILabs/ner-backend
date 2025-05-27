# PocketShield

This is an Electron application for Named Entity Recognition that migrates the functionality from the Next.js frontend app.

## Setup and Migration

### Install Dependencies

First, install the necessary dependencies:

```bash
npm install
```

### Run Migration Script

Run the migration script to copy components and assets from the Next.js app:

```bash
./migrate.sh
```

### Development Mode

To run the app in development mode:

```bash
npm run dev
```

This will start both the Vite development server and the Electron app that connects to it.

### Building for Production

To build the app for production:

```bash
npm run build
```

## Migration Notes

The migration process involves:

1. Setting up a Vite-based React development environment within the Electron app
2. Copying UI components, styles, and assets from the Next.js app
3. Adapting the routing system to work with Electron
4. Ensuring API calls work correctly from the Electron environment

### API Endpoints

API endpoints should be configured to point to the same backend as the Next.js app was using. Update the API URLs in the appropriate configuration files if needed.

### CSS and Styling

The migration preserves the Tailwind CSS styling from the original Next.js app.

### Component Adaptation

Some Next.js specific components may need adjustment to work in a standard React + Vite environment:

- Next.js `Link` components should be replaced with standard navigation
- Next.js specific routing needs to be adapted
- Server components must be converted to client components

## Troubleshooting

If you encounter issues:

1. Check the developer console for errors (View > Toggle Developer Tools)
2. Ensure all dependencies are installed
3. Verify that API endpoints are correctly configured 

## Running with Integrated Backend

This Electron app now has the Go backend integrated. Here's how to run it:

### Development Mode

1. Make sure you have the Go backend built:
   ```
   cd cmd/local/
   go clean -cache
   CGO_LDFLAGS="-L$(pwd)" go build -o main
   cp main ../../
   cd ../../electron-ner-app
   ```

2. Start the integrated app:
   ```
   MODEL_PATH=/share/pratik/onnx_model  npm run dev
   ```
   This will:
   - Copy the Go backend to the `bin` directory
   - Start the Vite development server
   - Start Electron
   - Start the Go backend

### Production Build (macOS)

To build a macOS DMG with integrated backend:

```
MODEL_PATH=/share/pratik/onnx_model npm run build
```

This will:
1. Build the Go backend for macOS
2. Copy it to the Electron app
3. Build the React frontend
4. Package everything into a DMG file with the backend included

The resulting DMG file will be in the `dist` directory. 

When installed, the app will automatically find and use the backend without any additional steps required.

### Troubleshooting

If you encounter issues with the backend not starting:

1. Check the app logs via the developer tools console (View > Toggle Developer Tools)
2. Ensure there are no firewall issues blocking port 8000
3. Make sure no other processes are using port 8000 