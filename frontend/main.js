const { app, BrowserWindow, ipcMain, protocol } = require('electron');
const path = require('path');
const url = require('url');
const fs = require('fs');
const isDev = process.env.NODE_ENV !== 'production';

// Keep a global reference of the window object to avoid it being garbage collected
let mainWindow;

function createWindow() {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false, // Recommended for security
      contextIsolation: true, // Recommended for security
      preload: path.join(__dirname, 'preload.js') // Preload script
    }
  });

  // Set up protocol for serving static files in production
  if (!isDev) {
    protocol.handle('file', (request) => {
      let pathname = new URL(request.url).pathname;
      
      // Handle paths that don't end with file extensions (like /about)
      // by serving the corresponding /about/index.html
      if (!path.extname(pathname)) {
        pathname = path.join(pathname, 'index.html');
      }
      
      return new Response(fs.readFileSync(path.join(__dirname, 'out', pathname)));
    });
  }

  // Determine the URL to load
  const startUrl = isDev 
    ? 'http://localhost:3007' // Development server URL
    : url.format({ // Production build
        pathname: path.join(__dirname, './out/index.html'),
        protocol: 'file:',
        slashes: true
      });
  
  // Load the app
  mainWindow.loadURL(startUrl);

  // Open DevTools in development mode
  if (isDev) {
    mainWindow.webContents.openDevTools();
  }

  // Emitted when the window is closed
  mainWindow.on('closed', () => {
    // Dereference the window object
    mainWindow = null;
  });
}

// Create window when Electron has finished initialization
app.whenReady().then(() => {
  createWindow();
  
  // Set up custom protocol
  if (!isDev) {
    protocol.registerFileProtocol('file', (request, callback) => {
      const pathname = decodeURI(request.url.replace('file:///', ''));
      callback(pathname);
    });
  }
});

// Quit when all windows are closed, except on macOS where it's common
// for applications to stay open until explicitly quit
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  // On macOS, re-create a window when dock icon is clicked and no other windows open
  if (mainWindow === null) {
    createWindow();
  }
}); 