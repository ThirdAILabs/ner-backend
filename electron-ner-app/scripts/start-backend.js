import { spawn } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { FIXED_PORT, ensurePortIsFree } from './check-port.js';
import log from 'electron-log';

log.transports.file.level = 'debug';
log.transports.file.resolvePath = () => {
  // same folder but name it backend.log
  return path.join(
    process.resourcesPath || __dirname,
    '..', // up to Resources
    'logs',
    'backend-launcher.log'
  );
};

log.debug('→ start-backend.js initializing…');


// Get __dirname equivalent in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Determine if we're in Electron or standalone
const isElectron = process.versions.hasOwnProperty('electron');

// This function determines the correct path to the bin directory
function getBinPath() {
  // Process environment and execution context
  const isPkg = 'electron' in process.versions;
  const isProduction = process.env.NODE_ENV === 'production';

  log.debug('Debug path info:');
  log.debug('- isPkg:', isPkg);
  log.debug('- isProduction:', isProduction);
  log.debug('- __dirname:', __dirname);
  log.debug('- process.resourcesPath:', process.resourcesPath);
  log.debug('- process.execPath:', process.execPath);
  log.debug('- Current working directory:', process.cwd());

  // Forced NODE_ENV override - ensure we're checking properly
  if (process.execPath && process.execPath.includes('Applications')) {
    log.debug('App appears to be installed in Applications - forcing production mode');
    process.env.NODE_ENV = 'production';
  }

  // In development mode
  if (!isProduction && !process.execPath.includes('Applications')) {
    // We'll use the executable from the bin directory in the project
    const devPath = path.join(__dirname, '..', 'bin');
    if (fs.existsSync(devPath)) {
      log.debug('Found bin directory:', devPath);
      return devPath;
    }

    // Fall back to looking in the parent directory (where the Go project is)
    const parentPath = path.join(__dirname, '..', '..');
    if (fs.existsSync(parentPath)) {
      log.debug('Found parent directory:', parentPath);
      return parentPath;
    }

    log.error('Bin directory not found in development mode');
    return null;
  }

  // In production or installed app - determine the correct path

  // In production, we ALWAYS want Resources/bin under the .app bundle
  const resourcesDir = process.resourcesPath
    || (process.platform === 'darwin'
      ? path.join(path.dirname(process.execPath), '..', 'Resources')
      : null);

  if (!resourcesDir) {
    log.error('Could not determine your app\'s Resources directory.');
    return null;
  }

  log.debug('Looking for bin under Resources at:', resourcesDir);
  const binDir = path.join(resourcesDir, 'bin');
  if (fs.existsSync(binDir)) {
    log.debug('Found bin directory at:', binDir);
    return binDir;
  }

  log.error('bin directory not found at:', binDir);

  // Try different approaches to find resources directory
  if (process.resourcesPath) {
    // Electron provides this directly in packaged apps
    resourcesDir = process.resourcesPath;
    log.debug('Using process.resourcesPath:', resourcesDir);
  } else if (process.execPath && process.platform === 'darwin') {
    // On macOS, resources are in a standard location relative to the executable
    resourcesDir = path.join(path.dirname(process.execPath), '..', 'Resources');
    log.debug('Using macOS standard path:', resourcesDir);
  } else {
    // Try to infer from __dirname
    resourcesDir = path.join(__dirname, '..', '..', '..');
    if (path.basename(resourcesDir) !== 'Resources') {
      resourcesDir = path.join(resourcesDir, 'Resources');
    }
    log.debug('Inferred Resources path:', resourcesDir);
  }

  // Primary location for packaged apps with absolute path for GUI launches
  const primaryLocations = [
    path.join(resourcesDir, 'bin'),
    '/Applications/PocketShield.app/Contents/Resources/bin'
  ];

  for (const primaryLocation of primaryLocations) {
    log.debug('Checking primary location:', primaryLocation);
    if (fs.existsSync(primaryLocation)) {
      log.debug('✅ Found bin at primary location');
      return primaryLocation;
    }
  }

  // List of fallback locations
  const fallbackPaths = [
    path.join(resourcesDir, 'Resources', 'bin'),
    path.join(resourcesDir, 'app.asar.unpacked', 'bin'),
    path.join(resourcesDir, '..', 'bin'),
    // Path with current execPath directory (absolute)
    path.join(path.dirname(process.execPath || ''), '..', 'Resources', 'bin')
  ];

  // Check each possible path
  for (const possiblePath of fallbackPaths) {
    log.debug('Checking fallback path:', possiblePath);
    if (fs.existsSync(possiblePath)) {
      log.debug('✅ Found bin at fallback path:', possiblePath);
      return possiblePath;
    }
  }

  // Last resort - search common paths
  log.debug('Searching standard macOS application paths...');
  const standardMacPaths = [
    '/Applications/PocketShield.app/Contents/Resources/bin',
    '/Applications/PocketShield.app/Contents/MacOS/bin',
    path.join(process.env.HOME || '', 'Applications/PocketShield.app/Contents/Resources/bin')
  ];

  for (const stdPath of standardMacPaths) {
    log.debug('Checking standard path:', stdPath);
    if (fs.existsSync(stdPath)) {
      log.debug('✅ Found bin at standard path:', stdPath);
      return stdPath;
    }
  }

  // Last resort - try to find and list the resources directory contents
  try {
    log.debug('Contents of resources directory:');
    if (fs.existsSync(resourcesDir)) {
      const contents = fs.readdirSync(resourcesDir);
      log.debug(contents);

      // Check if bin directory exists
      const binDir = path.join(resourcesDir, 'bin');
      if (fs.existsSync(binDir)) {
        log.debug('Contents of bin directory:');
        log.debug(fs.readdirSync(binDir));
      }
    }
  } catch (error) {
    log.error('Error listing directory contents:', error.message);
  }

  log.error('❌ Could not find bin directory in any location');
  return null;
}

// Get the path to the backend executable
function getBackendPath() {
  const binPath = getBinPath();
  if (!binPath) return null;
  return path.join(binPath, 'main');
}

// Get the path to the default model file
function getDefaultModelPath() {
  const binPath = getBinPath();
  if (!binPath) return null;
  return path.join(binPath, 'udt_complete.model');
}

// In start-backend.js, replace the startBackend function

export async function startBackend() {
  // Ensure our fixed port is available
  const portAvailable = await ensurePortIsFree();
  if (!portAvailable) {
    log.error(`🔴 Could not secure port ${FIXED_PORT} for backend use`);
    return null;
  }

  log.debug(`Using fixed port: ${FIXED_PORT}`)

  const backendPath = getBackendPath();
  const backendDir = path.dirname(backendPath);

  // Check if backend executable exists and is executable
  if (!fs.existsSync(backendPath)) {
    log.error(`Backend executable not found at: ${backendPath}`);
    return null;
  }

  try {
    const stats = fs.statSync(backendPath);
    if (!(stats.mode & parseInt('111', 8))) {
      log.error(`Backend executable is not executable: ${backendPath}`);
      return null;
    }
  } catch (error) {
    log.error(`Cannot check backend executable permissions: ${error.message}`);
    return null;
  }

  log.debug('Spawning backend with cwd:', backendDir);

  let backendLogPath = path.join(
    process.resourcesPath || __dirname,
    '..', // up to Resources
    'logs',
    'ner-backend.log'
  )

  // Ensure logs directory exists
  const logsDir = path.dirname(backendLogPath);
  if (!fs.existsSync(logsDir)) {
    fs.mkdirSync(logsDir, { recursive: true });
  }

  const outFd = fs.openSync(backendLogPath, 'a')
  const errFd = fs.openSync(backendLogPath, 'a')

  log.debug(`All backend output → ${backendLogPath}`)

  const proc = spawn(
    backendPath,
    [], {
    cwd: backendDir,
    env: {
      ...process.env,
      PORT: FIXED_PORT.toString(),
      MODEL_PATH: getDefaultModelPath(),
    },
    stdio: ['ignore', outFd, errFd]
  })

  let backendStarted = false;
  let startupError = null;

  proc.on('error', err => {
    log.error('Backend spawn error:', err)
    startupError = err;
  })

  let fdsClosed = false;
  function closeFds() {
    if (fdsClosed) return;
    fdsClosed = true;

    try {
      fs.closeSync(outFd);
    } catch (err) {
      if (err.code !== 'EBADF') throw err;
    }

    try {
      fs.closeSync(errFd);
    } catch (err) {
      if (err.code !== 'EBADF') throw err;
    }
  }

  proc.on('exit', (code, signal) => {
    log.debug(`Backend exit — code: ${code}, signal: ${signal}`);
    closeFds();
    
    if (code !== 0 && code !== null && !backendStarted) {
      startupError = new Error(`Backend exited with code ${code}`);
    }
  });

  proc.on('close', code => {
    log.debug(`Backend stdio closed, code ${code}`);
    if (code !== 0 && code !== null) {
      log.error(`Backend exited with code ${code}`);
      if (!backendStarted) {
        startupError = new Error(`Backend closed with code ${code}`);
      }
    }
    closeFds();
  });

  process.on('SIGINT', () => {
    log.debug('Received SIGINT, stopping backend…')
    proc.kill('SIGINT')
  })
  process.on('SIGTERM', () => {
    log.debug('Received SIGTERM, stopping backend…')
    proc.kill('SIGTERM')
  })

  // Wait longer and check if backend is actually running
  await new Promise(resolve => setTimeout(resolve, 2000));

  // Check if process is still running and no startup error occurred
  if (startupError) {
    log.error('Backend failed to start:', startupError);
    return null;
  }

  if (proc.killed || proc.exitCode !== null) {
    log.error('Backend process terminated during startup');
    return null;
  }

  // Try to verify the backend is responding (optional)
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
      backendStarted = true; // Assume it's working if process is alive
    }
  } catch (error) {
    log.warn('Backend health check failed:', error.message);
    // Still assume it's working if process is alive
    backendStarted = true;
  }

  return { 
    process: proc, 
    kill: sig => proc.kill(sig),
    isRunning: () => !proc.killed && proc.exitCode === null
  }
}

// Start the backend if this script is called directly
if (import.meta.url === import.meta.main) {
  startBackend();
} 