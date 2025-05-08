const { app, BrowserWindow } = require('electron');
const path = require('path');
const isDev = require('electron-is-dev');
const { startBackend } = require('./scripts/start-backend');

// Keep a global reference of the window object to prevent it from being garbage collected
let mainWindow;
let backendProcess = null;
let backendStarted = false;

function createWindow() {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,  // Enable context isolation for security
      preload: path.join(__dirname, 'preload.js')
    }
  });

  if (isDev) {
    // In development mode, add a delay to ensure Vite has time to start
    console.log("Development mode: Waiting for Vite server to start...");
    setTimeout(() => {
      console.log("Attempting to load from Vite dev server...");
      mainWindow.loadURL('http://localhost:3007/');
      // Open DevTools in development
      mainWindow.webContents.openDevTools();
    }, 3000); // 3 second delay
  } else {
    console.log("Production mode: Loading built app");
    // Path to the built HTML file
    const htmlPath = path.join(__dirname, 'src/dist/index.html');
    console.log("Loading HTML from:", htmlPath);
    
    // Load built app in production
    mainWindow.loadFile(htmlPath);
    
    // Uncomment to open DevTools in production for debugging
    mainWindow.webContents.openDevTools();
  }

  // Emitted when the window is closed
  mainWindow.on('closed', () => {
    // Dereference the window object
    mainWindow = null;
  });
}

// Start backend and return the process
function ensureBackendStarted() {
  if (!backendStarted) {
    console.log('Starting backend...');
    try {
      backendProcess = startBackend();
      if (backendProcess) {
        console.log('Backend started successfully');
        backendStarted = true;
      } else {
        console.error('Failed to start backend process');
      }
    } catch (error) {
      console.error('Error starting backend:', error);
    }
  }
  return backendProcess;
}

// This method will be called when Electron has finished initialization
app.whenReady().then(() => {
  // Always start the backend first, regardless of dev/prod mode
  ensureBackendStarted();
  createWindow();

  app.on('activate', () => {
    // On macOS it's common to re-create a window when the dock icon is clicked
    if (mainWindow === null) createWindow();
  });
});

// Quit when all windows are closed, except on macOS
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// Clean up backend on app quit
app.on('will-quit', () => {
  console.log('App is quitting, cleaning up backend...');
  if (backendProcess) {
    backendProcess.kill('SIGINT');
  }
}); 