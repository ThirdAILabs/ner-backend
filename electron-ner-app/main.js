import { app, BrowserWindow, dialog, ipcMain, shell } from 'electron';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import serve from 'electron-serve';
import { startBackend } from './scripts/start-backend.js';
import { openFileChooser, openFile, showFileInFolder } from './scripts/file-utils.js';
import { initTelemetry, insertTelemetryEvent, closeTelemetry } from './telemetry.js';
import { initializeUserId, getCurrentUserId } from './userIdManager.js';
import axios from 'axios';
import FormData from 'form-data';

import log from 'electron-log';
import electronUpdater from 'electron-updater';

const { autoUpdater } = electronUpdater;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const pendingUpdateFile = path.join(app.getPath('userData'), 'pending-update.json');
const installedUpdateFile = path.join(app.getPath('userData'), 'update-installed.json');

const appServe = app.isPackaged ? serve({
    directory: path.join(__dirname, "frontend-dist")
}) : null;

let isDev = false;
try {
    const electronIsDev = await import('electron-is-dev');
    isDev = electronIsDev.default;
} catch (error) {
    console.warn('electron-is-dev module not found, assuming production mode');
    isDev = false;
}

if (!isDev) {
    process.env.NODE_ENV = 'production';
    console.log('Setting NODE_ENV to production');
    try {
        const binPath = path.join(process.resourcesPath, 'bin');
        if (fs.existsSync(binPath)) {
            process.chdir(process.resourcesPath);
            console.log('Changed working directory to:', process.cwd());
        }
    } catch (e) {
        log.error('Failed to set working directory:', e);
    }
}

let mainWindow;
let backendProcess = null;
let backendStarted = false;
let updateHandled = false;
let isUpdateInstall = false;

ipcMain.handle('open-external-link', async (_, url) => await shell.openExternal(url));

// Window control handlers
ipcMain.handle('minimize-window', () => {
    if (mainWindow) mainWindow.minimize();
});

ipcMain.handle('maximize-window', () => {
    if (mainWindow) {
        if (mainWindow.isMaximized()) {
            mainWindow.unmaximize();
        } else {
            mainWindow.maximize();
        }
    }
});

ipcMain.handle('close-window', () => {
    if (mainWindow) mainWindow.close();
});

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        frame: false,
        titleBarStyle: 'hiddenInset',
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js')
        }
    });

    if (isDev) {
        console.log("Development mode: Waiting for Next.js server to start...");
        setTimeout(() => mainWindow.loadURL('http://localhost:3007/'), 3000);
    } else {
        console.log("Production mode: Loading built app");
        appServe(mainWindow).then(() => mainWindow.loadURL("app://-"));
    }

    mainWindow.on('closed', () => { mainWindow = null; });
}

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
                log.error(errorMsg);
                if (mainWindow) {
                    dialog.showErrorBox('Backend Error',
                        `Failed to start the backend service. The application may not function correctly.
\nPlease check the installation.`);
                }
            }
        } catch (error) {
            log.error('Error starting backend:', error);
            if (mainWindow) {
                dialog.showErrorBox('Backend Error', `Error starting backend service: ${error.message}`);
            }
        }
    }
    return backendProcess;
}

autoUpdater.logger = log;
autoUpdater.logger.transports.file.level = 'info';
autoUpdater.autoDownload = false;
autoUpdater.autoInstallOnAppQuit = false;

// !!!! COMMENT OUT THE FOLLWOWING LINE IN PRODUCTION !!!
// autoUpdater.allowPrerelease = true;

autoUpdater.on('checking-for-update', () => console.log('Checking for update...'));

autoUpdater.on('update-available', async (info) => {
    console.log('Update available:', info);
    const result = await dialog.showMessageBox(mainWindow, {
        type: 'question',
        buttons: ['Download Now', 'Remind Me Later'],
        defaultId: 0,
        cancelId: 1,
        title: 'Update Available',
        message: `Version ${info.version} is available.`
    });

    if (result.response === 0) {
        autoUpdater.downloadUpdate();
    } else {
        fs.writeFileSync(pendingUpdateFile, JSON.stringify({ version: info.version }), 'utf-8');
    }
});

autoUpdater.on('update-downloaded', async (info) => {
    if (updateHandled) return;
    updateHandled = true;

    if (isUpdateInstall) {
        // Will this ever hit??
        await shutdownBackend();
        // setTimeout(() => app.exit(0), 1000);
        autoUpdater.quitAndInstall();
        return;
    }

    const result = await dialog.showMessageBox(mainWindow, {
        type: 'question',
        buttons: ['Install & Restart Now', 'Later'],
        defaultId: 0,
        cancelId: 1,
        title: 'Update Ready',
        message: `Version ${info.version} is ready to install.`
    });

    if (result.response === 0) {
        isUpdateInstall = true;
        await shutdownBackend();
        autoUpdater.quitAndInstall();
    } else {
        fs.writeFileSync(pendingUpdateFile, JSON.stringify({ version: info.version }), 'utf-8');
    }
});

autoUpdater.on('error', (err) => log.error('Error in auto-updater:', err));

autoUpdater.on('download-progress', (progress) => {
    console.log(`Download speed: ${progress.bytesPerSecond} - Downloaded ${Math.round(progress.percent)}%`);
});

ipcMain.handle('telemetry', async (_, data) => await insertTelemetryEvent(data));
ipcMain.handle('get-user-id', async () => getCurrentUserId());
ipcMain.handle('open-file-chooser', async (_, types, isDirectoryMode, isCombinedMode) => openFileChooser(types, isDirectoryMode, isCombinedMode));
ipcMain.handle('open-file', async (_, filePath) => openFile(filePath));
ipcMain.handle('show-file-in-folder', async (_, filePath) => { showFileInFolder(filePath); });
ipcMain.handle('upload-files', async (event, { filePaths, uploadUrl, uniqueNames, originalNames }) => {
    // This function handles the actual uploading of files to the backend.
    // For the UI, we only store the file metadata in the memory to display to the users.
    // When the user submits the report, we upload the files to the backend.
    const form = new FormData();

    for (let i = 0; i < filePaths.length; i++) {
        const filePath = filePaths[i];
        // Use the unique name for upload
        const filename = uniqueNames && uniqueNames[i] ? uniqueNames[i] : path.basename(filePath);
        form.append('files', fs.createReadStream(filePath), { filename });
    }

    const response = await axios.post(uploadUrl, form, {
        headers: form.getHeaders()
    });

    return { success: true, uploadId: response.data.Id };
});


app.whenReady().then(async () => {
    await initializeUserId();
    await initTelemetry();
    await ensureBackendStarted();
    createWindow();

    const currentVersion = app.getVersion();

    if (fs.existsSync(installedUpdateFile)) {
        try {
            const { version } = JSON.parse(fs.readFileSync(installedUpdateFile, 'utf-8'));
            if (version === currentVersion) {
                fs.unlinkSync(installedUpdateFile);
                updateHandled = true;
            }
        } catch (e) {
            try { fs.unlinkSync(installedUpdateFile); } catch (_) { }
        }
    }

    if (fs.existsSync(pendingUpdateFile)) {
        try {
            dialog.showMessageBox({ type: 'info', title: 'Update available', message: `A new version (${info.version}) is available.` });

            // const { version } = JSON.parse(fs.readFileSync(pendingUpdateFile, 'utf-8'));
            // if (version && version !== currentVersion) {
            //     const reminder = await dialog.showMessageBox(mainWindow, {
            //         type: 'question',
            //         buttons: ['Download Now', 'Later'],
            //         defaultId: 0,
            //         cancelId: 1,
            //         title: 'Update Ready',
            //         message: `Version ${version} is available.`
            //     });
            //     if (reminder.response === 0) {
            //         isUpdateInstall = true;
            //         autoUpdater.downloadUpdate();
            //     }
            // }
        } catch (e) {
            console.error('Error reading pending-update file:', e);
        } finally {
            try { fs.unlinkSync(pendingUpdateFile); } catch (_) { }
        }
    }

    if (!fs.existsSync(pendingUpdateFile) && (!isDev || process.env.DEBUG_UPDATER === 'true')) {
        autoUpdater.checkForUpdatesAndNotify()
            .then(result => console.log('checkForUpdatesAndNotify result:', result))
            .catch(error => log.error('checkForUpdatesAndNotify error:', error));
    }

    app.on('activate', () => {
        if (mainWindow === null) createWindow();
    });
});

async function shutdownBackend() {
    if (backendProcess?.kill) {
        backendProcess.kill('SIGTERM');
        await new Promise(resolve => {
            const timeout = setTimeout(() => {
                backendProcess.kill('SIGKILL');
                resolve();
            }, 3000);
            backendProcess.process?.on('exit', () => {
                clearTimeout(timeout);
                resolve();
            });
        });
    }
}

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
});

app.on('will-quit', () => {
    if (backendProcess?.kill) {
        backendProcess.kill('SIGINT');
    }
});

app.on('quit', async () => {
    if (backendProcess?.kill) {
        backendProcess.kill('SIGTERM');
    }
    await closeTelemetry();
});
