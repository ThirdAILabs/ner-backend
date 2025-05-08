# NER Electron App

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
   cd ..
   go build -o main
   cd electron-ner-app
   ```

2. Start the integrated app:
   ```
   npm run dev
   ```
   This will:
   - Copy the Go backend to the `bin` directory
   - Start the Vite development server
   - Start Electron
   - Start the Go backend

### Production Build (macOS)

To build a macOS DMG with integrated backend:

```
npm run build-dmg
```

This will:
1. Build the Go backend for macOS
2. Copy it to the Electron app
3. Build the React frontend
4. Package everything into a DMG file

The resulting DMG file will be in the `dist` directory.

### Installing the App

When you install the app from the DMG, you might need to run an additional script to ensure the backend is correctly installed:

1. Install the app by dragging it to the Applications folder
2. Open Terminal and run the installation script:
   ```
   cd path/to/electron-ner-app/dist
   ./install-backend.sh
   ```
   
This script will:
- Create the necessary `bin` directory in the app
- Copy the backend binary to the correct location
- Set the appropriate permissions

### Troubleshooting

If you encounter issues with the backend not starting:

1. Check that the Go backend executable exists and is executable:
   ```
   ls -la /Applications/NER\ Electron\ App.app/Contents/Resources/bin/main
   ```

2. If the file is missing, run the installation script:
   ```
   cd path/to/electron-ner-app/dist
   ./install-backend.sh
   ```

3. Check the Electron logs for any errors when starting the backend 