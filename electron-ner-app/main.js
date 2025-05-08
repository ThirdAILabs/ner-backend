const { app, BrowserWindow } = require('electron');
const path = require('path');
const isDev = require('electron-is-dev');
const { startBackend } = require('./scripts/start-backend');

// Keep a global reference of the window object to prevent it from being garbage collected
let mainWindow;
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
    // Load built app in production
    mainWindow.loadFile(path.join(__dirname, 'src/dist/index.html'));
  }

  // Emitted when the window is closed
  mainWindow.on('closed', () => {
    // Dereference the window object
    mainWindow = null;
  });
}

// Start backend if running in production
// In development, backend is started via npm script
function ensureBackendStarted() {
  if (!backendStarted && !isDev) {
    console.log('Starting backend in production mode...');
    startBackend();
    backendStarted = true;
  }
}

// This method will be called when Electron has finished initialization
app.whenReady().then(() => {
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