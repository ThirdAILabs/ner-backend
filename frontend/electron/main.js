const { app, BrowserWindow, ipcMain } = require('electron');
const serve = require('electron-serve');
const path = require('path');
const fs = require('fs');

const isProd = process.env.NODE_ENV === 'production';

let mainWindow;

const loadURL = serve({ directory: 'out' });

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      preload: path.join(__dirname, 'preload.js')
    }
  });

  if (isProd) {
    loadURL(mainWindow);
  } else {
    mainWindow.loadURL('http://localhost:3007');
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// IPC handlers for communication between renderer and main process
ipcMain.handle('get-deployment-ids', async () => {
  // In a real app, this would fetch from your backend
  // For now, we'll return some mock data
  return ['deployment1', 'deployment2', 'deployment3'];
}); 