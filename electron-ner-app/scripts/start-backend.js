import { spawn } from 'node:child_process';
import { app } from 'electron';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { FIXED_PORT, ensurePortIsFree } from './check-port.js';
import log from 'electron-log';
import { get } from 'node:http';

log.transports.file.level = 'debug';
log.transports.file.resolvePath = () => {
  return path.join(
    process.resourcesPath || __dirname,
    '..',
    'logs',
    'backend-launcher.log'
  );
};

log.debug('→ start-backend.js initializing…');

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const isElectron = process.versions.hasOwnProperty('electron');

function getBinPath() {
  const isProduction = process.env.NODE_ENV === 'production';

  log.debug('Debug path info:');
  log.debug('- isProduction:', isProduction);
  log.debug('- __dirname:', __dirname);
  log.debug('- process.resourcesPath:', process.resourcesPath);
  log.debug('- process.execPath:', process.execPath);
  log.debug('- Current working directory:', process.cwd());

  if (process.execPath && process.execPath.includes('Applications')) {
    log.debug('App appears to be installed in Applications - forcing production mode');
    process.env.NODE_ENV = 'production';
  }

  if (!isProduction && !process.execPath.includes('Applications')) {
    const devPath = path.join(__dirname, '..', 'bin');
    if (fs.existsSync(devPath)) return devPath;
    const parentPath = path.join(__dirname, '..', '..');
    if (fs.existsSync(parentPath)) return parentPath;
    return null;
  }

  const resourcesDir = process.resourcesPath || (process.platform === 'darwin'
    ? path.join(path.dirname(process.execPath), '..', 'Resources')
    : null);

  if (!resourcesDir) return null;

  const binDir = path.join(resourcesDir, 'bin');
  if (fs.existsSync(binDir)) return binDir;

  const fallbackPaths = [
    path.join(resourcesDir, 'Resources', 'bin'),
    path.join(resourcesDir, 'app.asar.unpacked', 'bin'),
    path.join(resourcesDir, '..', 'bin'),
    path.join(path.dirname(process.execPath || ''), '..', 'Resources', 'bin')
  ];

  for (const possiblePath of fallbackPaths) {
    if (fs.existsSync(possiblePath)) return possiblePath;
  }

  const standardMacPaths = [
    '/Applications/PocketShield.app/Contents/Resources/bin',
    '/Applications/PocketShield.app/Contents/MacOS/bin',
    path.join(process.env.HOME || '', 'Applications/PocketShield.app/Contents/Resources/bin')
  ];

  for (const stdPath of standardMacPaths) {
    if (fs.existsSync(stdPath)) return stdPath;
  }

  return null;
}

function getBackendPath() {
  const binPath = getBinPath();
  if (!binPath) return null;
  
  // Platform-specific executable name
  const executableName = process.platform === 'win32' ? 'main.exe' : 'main';
  return path.join(binPath, executableName);
}

function getModelConfigPath() {
  const binPath = getBinPath();
  if (!binPath) return null;
  return path.join(binPath, 'model_config.json');
}

export async function startBackend() {
  const portAvailable = await ensurePortIsFree();
  if (!portAvailable) {
    log.error(`🔴 Could not secure port ${FIXED_PORT} for backend use`);
    return null;
  }

  log.debug(`Using fixed port: ${FIXED_PORT}`);

  const backendPath = getBackendPath();
  const backendDir  = path.dirname(backendPath);

  await app.whenReady();
  const appDataDir = app.getPath('userData');
  const modelConfigPath = getModelConfigPath();

  if (!fs.existsSync(backendPath)) {
    log.error(`Backend executable not found at: ${backendPath}`);
    return null;
  }

  try {
    // Platform-specific executable permission check
    if (process.platform === 'win32') {
      // On Windows, just check if file exists
      fs.accessSync(backendPath, fs.constants.F_OK);
    } else {
      // On Unix-like systems, check execute permission
      const stats = fs.statSync(backendPath);
      if (!(stats.mode & parseInt('111', 8))) {
        log.error(`Backend executable is not executable: ${backendPath}`);
        return null;
      }
    }
  } catch (error) {
    log.error(`Cannot check backend executable permissions: ${error.message}`);
    return null;
  }

  log.debug('Spawning backend with cwd:', backendDir);

  let modelType = 'bolt_udt';  // Default model type
  try {
    const modelConfig = JSON.parse(fs.readFileSync(modelConfigPath, 'utf8'));
    modelType = modelConfig.model_type || modelType;
    log.debug('Loaded model type from config:', modelType);
  } catch (error) {
    log.warn('Failed to load model config, using default:', error.message);
  }

  let backendLogPath = path.join(
    process.resourcesPath || __dirname,
    '..',
    'logs',
    'ner-backend.log'
  );

  if (!fs.existsSync(path.dirname(backendLogPath))) {
    fs.mkdirSync(path.dirname(backendLogPath), { recursive: true });
  }

  log.debug(`All backend output → ${backendLogPath}`);

  // Get the plugin executable path
  const pluginPath = path.join(backendDir, 'plugin', 'plugin');

  const frameworksDir = path.join(
    process.resourcesPath || __dirname,
    '..', // up to Resources
    'Frameworks');
    
  const proc = spawn(
    backendPath,
    [],
    {
      cwd: backendDir,
      env: {
        PORT:       FIXED_PORT.toString(),
        MODEL_DIR: getBinPath(),
        MODEL_TYPE: modelType,
        PLUGIN_SERVER: pluginPath,
        APP_DATA_DIR: appDataDir,
        ONNX_RUNTIME_DYLIB: process.platform === 'win32' 
          ? path.join(getBinPath(), 'onnxruntime.dll')
          : (frameworksDir ? path.join(frameworksDir, 'libonnxruntime.dylib') : ''),
      },
      stdio: ['pipe', 'pipe', 'pipe']
    }
  );

  const logStream = fs.createWriteStream(backendLogPath, { flags: 'a' });
  let logStreamEnded = false;
  function safeEndLogStream() {
    if (!logStreamEnded) {
      logStreamEnded = true;
      logStream.end();
    }
  }

  proc.stdout.pipe(logStream);
  proc.stderr.pipe(logStream);

  let backendStarted = false;
  let startupError = null;

  proc.on('error', err => {
    log.error('Backend spawn error:', err);
    startupError = err;
  });

  proc.on('exit', (code, signal) => {
    log.debug(`Backend exit — code: ${code}, signal: ${signal}`);
    safeEndLogStream();
    if (code !== 0 && code !== null && !backendStarted) {
      startupError = new Error(`Backend exited with code ${code}`);
    }
  });

  proc.on('close', code => {
    log.debug(`Backend stdio closed, code ${code}`);
    safeEndLogStream();
    if (code !== 0 && code !== null) {
      log.error(`Backend exited with code ${code}`);
      if (!backendStarted) {
        startupError = new Error(`Backend closed with code ${code}`);
      }
    }
  });

  process.on('SIGINT', () => {
    log.debug('Received SIGINT, stopping backend…');
    proc.kill('SIGINT');
  });

  process.on('SIGTERM', () => {
    log.debug('Received SIGTERM, stopping backend…');
    proc.kill('SIGTERM');
  });

  await new Promise(resolve => setTimeout(resolve, 2000));

  if (startupError) {
    log.error('Backend failed to start:', startupError);
    return null;
  }

  if (proc.killed || proc.exitCode !== null) {
    log.error('Backend process terminated during startup');
    return null;
  }

  try {
    const response = await fetch(`http://localhost:${FIXED_PORT}/health`, {
      method: 'GET',
      timeout: 5000
    });
    if (response.ok) {
      log.debug('Backend health check passed');
      backendStarted = true;
    } else {
      log.warn('Backend health check failed, but process is running');
      backendStarted = true;
    }
  } catch (error) {
    log.warn('Backend health check failed:', error.message);
    backendStarted = true;
  }

  return {
    process: proc,
    kill: sig => proc.kill(sig),
    isRunning: () => !proc.killed && proc.exitCode === null
  };
}

if (import.meta.url === import.meta.main) {
  startBackend();
}
