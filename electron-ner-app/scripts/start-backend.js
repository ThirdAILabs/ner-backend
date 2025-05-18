import { spawn } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import portfinder from 'portfinder';
import electron from 'electron';
import { app } from 'electron';
import { FIXED_PORT, ensurePortIsFree } from './check-port.js';
import log from 'electron-log';

log.transports.file.level = 'debug';
log.transports.file.resolvePath = () => {
  // same folder but name it backend.log
  return path.join(
    process.resourcesPath || __dirname,
    '..', // up to Resources
    'logs',
    'backend.log'
  );
};

log.info('→ start-backend.js initializing…');


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
  
  log.info('Debug path info:');
  log.info('- isPkg:', isPkg);
  log.info('- isProduction:', isProduction);
  log.info('- __dirname:', __dirname);
  log.info('- process.resourcesPath:', process.resourcesPath);
  log.info('- process.execPath:', process.execPath);
  log.info('- Current working directory:', process.cwd());
  
  // Forced NODE_ENV override - ensure we're checking properly
  if (process.execPath && process.execPath.includes('Applications')) {
    log.info('App appears to be installed in Applications - forcing production mode');
    process.env.NODE_ENV = 'production';
  }
  
  // In development mode
  if (!isProduction && !process.execPath.includes('Applications')) {
    // We'll use the executable from the bin directory in the project
    const devPath = path.join(__dirname, '..', 'bin');
    if (fs.existsSync(devPath)) {
      log.info('Found bin directory:', devPath);
      return devPath;
    }
    
    // Fall back to looking in the parent directory (where the Go project is)
    const parentPath = path.join(__dirname, '..', '..');
    if (fs.existsSync(parentPath)) {
      log.info('Found parent directory:', parentPath);
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

  log.info('Looking for bin under Resources at:', resourcesDir);
  const binDir = path.join(resourcesDir, 'bin');
  if (fs.existsSync(binDir)) {
    log.info('Found bin directory at:', binDir);
    return binDir;
  }

  log.error('bin directory not found at:', binDir);
  
  // Try different approaches to find resources directory
  if (process.resourcesPath) {
    // Electron provides this directly in packaged apps
    resourcesDir = process.resourcesPath;
    log.info('Using process.resourcesPath:', resourcesDir);
  } else if (process.execPath && process.platform === 'darwin') {
    // On macOS, resources are in a standard location relative to the executable
    resourcesDir = path.join(path.dirname(process.execPath), '..', 'Resources');
    log.info('Using macOS standard path:', resourcesDir);
  } else {
    // Try to infer from __dirname
    resourcesDir = path.join(__dirname, '..', '..', '..');
    if (path.basename(resourcesDir) !== 'Resources') {
      resourcesDir = path.join(resourcesDir, 'Resources');
    }
    log.info('Inferred Resources path:', resourcesDir);
  }
  
  // Primary location for packaged apps with absolute path for GUI launches
  const primaryLocations = [
    path.join(resourcesDir, 'bin'),
    '/Applications/PocketShield.app/Contents/Resources/bin'
  ];
  
  for (const primaryLocation of primaryLocations) {
    log.info('Checking primary location:', primaryLocation);
    if (fs.existsSync(primaryLocation)) {
      log.info('✅ Found bin at primary location');
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
    log.info('Checking fallback path:', possiblePath);
    if (fs.existsSync(possiblePath)) {
      log.info('✅ Found bin at fallback path:', possiblePath);
      return possiblePath;
    }
  }
  
  // Last resort - search common paths
  log.info('Searching standard macOS application paths...');
  const standardMacPaths = [
    '/Applications/PocketShield.app/Contents/Resources/bin',
    '/Applications/PocketShield.app/Contents/MacOS/bin',
    path.join(process.env.HOME || '', 'Applications/PocketShield.app/Contents/Resources/bin')
  ];
  
  for (const stdPath of standardMacPaths) {
    log.info('Checking standard path:', stdPath);
    if (fs.existsSync(stdPath)) {
      log.info('✅ Found bin at standard path:', stdPath);
      return stdPath;
    }
  }
  
  // Last resort - try to find and list the resources directory contents
  try {
    log.info('Contents of resources directory:');
    if (fs.existsSync(resourcesDir)) {
      const contents = fs.readdirSync(resourcesDir);
      log.info(contents);
      
      // Check if bin directory exists
      const binDir = path.join(resourcesDir, 'bin');
      if (fs.existsSync(binDir)) {
        log.info('Contents of bin directory:');
        log.info(fs.readdirSync(binDir));
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

export async function startBackend() {
  // Ensure our fixed port is available
  const portAvailable = await ensurePortIsFree();
  if (!portAvailable) {
    log.error(`🔴 Could not secure port ${FIXED_PORT} for backend use`);
    return null;
  }

  log.info(`Using fixed port: ${FIXED_PORT}`)

  const resourcesDir = process.resourcesPath;
  const binDir       = path.join(resourcesDir, 'bin');

  // DUMP BIN DIR CONTENTS
  try {
    const files = fs.readdirSync(binDir);
    log.info(`Contents of ${binDir}:`, files);
  } catch (err) {
    log.error(`Could not list ${binDir}:`, err);
  }

  const backendPath = getBackendPath();
  const backendDir  = path.dirname(backendPath);

  log.info(`Checking backendPath: ${backendPath}`);
  log.info('  exists:', fs.existsSync(backendPath));
  if (fs.existsSync(backendPath)) {
    const st = fs.statSync(backendPath);
    log.info('  mode:', (st.mode & 0o777).toString(8));
    log.info('  size:', st.size);
  }

  log.info('Spawning backend with cwd:', backendDir);

  let backendLogPath = path.join(
    process.resourcesPath || __dirname,
    '..', // up to Resources
    'logs',
    'backend_go.log'
  )
  const outFd = fs.openSync(backendLogPath, 'a')
  const errFd = fs.openSync(backendLogPath, 'a')

  log.info(`All backend output → ${backendLogPath}`)

  const proc = spawn(
    backendPath,
    [], {
      cwd:    backendDir,
      env:    {
        ...process.env,
        PORT:       FIXED_PORT.toString(),
        MODEL_PATH: getDefaultModelPath(),
      },
      stdio:  ['ignore', outFd, errFd]  // <— attach child stdout → outFd, stderr → errFd
    }
  )

  proc.on('error', err => {
    log.error('Backend spawn error:', err)
  })
  proc.on('exit', (code, signal) => {
    log.info(`Backend exit — code: ${code}, signal: ${signal}`)
    // close our fds
    fs.closeSync(outFd)
    fs.closeSync(errFd)
  })

  // optional: give it a moment to bind
  await new Promise(r => setTimeout(r, 500))

  return { process: proc, kill: sig => proc.kill(sig) }
}


// Start the backend if this script is called directly
if (import.meta.url === import.meta.main) {
  startBackend();
} 