const { app, BrowserWindow, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const serve = require('electron-serve');
const { autoUpdater } = require('electron-updater');
const log = require('electron-log');

const appServe = app.isPackaged ? serve({
  directory: path.join(__dirname, "frontend-dist")
}) : null;

// Safely check if we're in development mode with fallback if module is missing
let isDev = false;
try {
  const electronIsDev = require('electron-is-dev');
  isDev = electronIsDev;
} catch (error) {
  console.warn('electron-is-dev module not found, assuming production mode');
  isDev = false;
}

const { startBackend } = require('./scripts/start-backend');

// Force NODE_ENV to 'production' when not in development mode
if (!isDev) {
  process.env.NODE_ENV = 'production';
  console.log('Setting NODE_ENV to production');
}

// Keep a global reference of the window object to prevent it from being garbage collected
let mainWindow;
let backendProcess = null;
let backendStarted = false;

// Ensure working directory is set to the app directory for backend
function fixWorkingDirectory() {
  // When app is launched by double-clicking, working directory could be wrong
  const appPath = app.getAppPath();
  const resourcesPath = process.resourcesPath || path.join(appPath, '..', '..');
  
  console.log('Current working directory:', process.cwd());
  console.log('App path:', appPath);
  console.log('Resources path:', resourcesPath);
  
  // If we're running in production and the binary doesn't exist, try to fix paths
  if (!isDev) {
    // Try to see if bin directory exists in resources
    const binPath = path.join(resourcesPath, 'bin');
    if (fs.existsSync(binPath)) {
      console.log('Binary directory found in resources:', binPath);
      // Change working directory to resources path for backend
      try {
        process.chdir(resourcesPath);
        console.log('Changed working directory to:', process.cwd());
      } catch (error) {
        console.error('Failed to change working directory:', error);
      }
    } else {
      console.warn('Binary directory not found in resources:', binPath);
    }
  }
}

function createWindow() {
  // Fix working directory before creating window
  fixWorkingDirectory();
  
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
    // In development mode, add a delay to ensure Next.js has time to start
    console.log("Development mode: Waiting for Next.js server to start...");
    setTimeout(() => {
      console.log("Attempting to load from Next.js server...");
      mainWindow.loadURL('http://localhost:3007/');
    }, 3000); // 3 second delay
  } else {
    console.log("Production mode: Loading built app");
    appServe(mainWindow).then(() => {
      console.log("Loaded app from serve");
      mainWindow.loadURL("app://-");
    });
  }

  // Emitted when the window is closed
  mainWindow.on('closed', () => {
    // Dereference the window object
    mainWindow = null;
  });
}

// Start backend and return the process
async function ensureBackendStarted() {
  if (!backendStarted) {
    console.log('Starting backend...');
    try {
      backendProcess = await startBackend();
      if (backendProcess) {
        console.log('Backend started successfully');
        backendStarted = true;
      } else {
        const errorMsg = 'Failed to start backend process. Backend executable not found.';
        console.error(errorMsg);
        
        // Show error dialog in GUI mode
        if (mainWindow) {
          dialog.showErrorBox('Backend Error', 
            `Failed to start the backend service. The application may not function correctly.
            
Please try running the install script:
sudo mkdir -p "/Applications/PocketShield.app/Contents/Resources/bin"
sudo cp "/path/to/main" "/Applications/PocketShield.app/Contents/Resources/bin/main"
sudo chmod 755 "/Applications/PocketShield.app/Contents/Resources/bin/main"`);
        }
      }
    } catch (error) {
      console.error('Error starting backend:', error);
      
      // Show error dialog in GUI mode
      if (mainWindow) {
        dialog.showErrorBox('Backend Error', 
          `Error starting backend service: ${error.message}
          
Please try running the install script from the terminal.`);
      }
    }
  }
  return backendProcess;
}

// Ensure logger is set up for auto-updater
autoUpdater.logger = log;
autoUpdater.logger.transports.file.level = 'info';
autoUpdater.autoDownload = true;

// Set up auto-updater event listeners
autoUpdater.on('checking-for-update', () => {
  log.info('Checking for update...');
});
autoUpdater.on('update-available', () => {
  dialog.showMessageBox({ type: 'info', title: 'Update available', message: 'A new version is available. Downloading now...' });
});
autoUpdater.on('update-not-available', () => {
  log.info('Update not available.');
});
autoUpdater.on('error', (err) => {
  log.error('Error in auto-updater:', err);
});
autoUpdater.on('download-progress', (progressObj) => {
  let log_message = "Download speed: " + progressObj.bytesPerSecond;
  log_message = log_message + ' - Downloaded ' + Math.round(progressObj.percent) + '%';
  log.info(log_message);
});
autoUpdater.on('update-downloaded', () => {
  dialog.showMessageBox({ title: 'Install updates', message: 'Updates downloaded, application will be quit for update...' })
    .then(() => {
      autoUpdater.quitAndInstall();
    });
});

// This method will be called when Electron has finished initialization
app.whenReady().then(() => {
  // Always start the backend first, regardless of dev/prod mode
  createWindow();
  ensureBackendStarted();
  // Check for updates on launch (in production only)
  if (!isDev) {
    autoUpdater.checkForUpdatesAndNotify();
  }

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
  if (backendProcess && backendProcess.kill) {
    console.log('Sending SIGINT to backend process...');
    backendProcess.kill('SIGINT');
  }
});

// Handle app quit
app.on('quit', () => {
  console.log('App is quitting, ensuring backend is cleaned up...');
  if (backendProcess && backendProcess.kill) {
    console.log('Sending SIGTERM to backend process...');
    backendProcess.kill('SIGTERM');
  }
}); 