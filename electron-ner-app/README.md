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