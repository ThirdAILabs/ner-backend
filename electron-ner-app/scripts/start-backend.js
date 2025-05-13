const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const portfinder = require('portfinder');
const electron = require('electron');

// Determine if we're in Electron or standalone
const isElectron = process.versions.hasOwnProperty('electron');

// This function determines the correct path to the backend executable
function getBackendPath() {
  // Process environment and execution context
  const isPkg = 'electron' in process.versions;
  const isProduction = process.env.NODE_ENV === 'production';
  
  console.log('Debug path info:');
  console.log('- isPkg:', isPkg);
  console.log('- isProduction:', isProduction);
  console.log('- __dirname:', __dirname);
  console.log('- process.resourcesPath:', process.resourcesPath);
  console.log('- process.execPath:', process.execPath);
  console.log('- Current working directory:', process.cwd());
  
  // Forced NODE_ENV override - ensure we're checking properly
  if (process.execPath && process.execPath.includes('Applications')) {
    console.log('App appears to be installed in Applications - forcing production mode');
    process.env.NODE_ENV = 'production';
  }
  
  // In development mode
  if (!isProduction && !process.execPath.includes('Applications')) {
    // We'll use the executable from the bin directory in the project
    const devPath = path.join(__dirname, '..', 'bin', 'main');
    if (fs.existsSync(devPath)) {
      console.log('Found backend in bin directory:', devPath);
      return devPath;
    }
    
    // Fall back to looking in the parent directory (where the Go project is)
    const parentPath = path.join(__dirname, '..', '..', 'main');
    if (fs.existsSync(parentPath)) {
      console.log('Found backend in parent directory:', parentPath);
      return parentPath;
    }
    
    console.error('Backend executable not found in development mode');
    return null;
  }
  
  // In production or installed app - determine the correct path

  // First check the current working directory for bin/main
  // This is after we've fixed the working directory in main.js
  const cwdBinPath = path.join(process.cwd(), 'bin', 'main');
  if (fs.existsSync(cwdBinPath)) {
    console.log('✅ Found backend in current working directory:', cwdBinPath);
    return cwdBinPath;
  }
  
  // Find the resources directory
  let resourcesDir;
  
  // Try different approaches to find resources directory
  if (process.resourcesPath) {
    // Electron provides this directly in packaged apps
    resourcesDir = process.resourcesPath;
    console.log('Using process.resourcesPath:', resourcesDir);
  } else if (process.execPath && process.platform === 'darwin') {
    // On macOS, resources are in a standard location relative to the executable
    resourcesDir = path.join(path.dirname(process.execPath), '..', 'Resources');
    console.log('Using macOS standard path:', resourcesDir);
  } else {
    // Try to infer from __dirname
    resourcesDir = path.join(__dirname, '..', '..', '..');
    if (path.basename(resourcesDir) !== 'Resources') {
      resourcesDir = path.join(resourcesDir, 'Resources');
    }
    console.log('Inferred Resources path:', resourcesDir);
  }
  
  // Primary location for packaged apps with absolute path for GUI launches
  const primaryLocations = [
    path.join(resourcesDir, 'bin', 'main'),
    '/Applications/PocketShield.app/Contents/Resources/bin/main'
  ];
  
  for (const primaryLocation of primaryLocations) {
    console.log('Checking primary location:', primaryLocation);
    if (fs.existsSync(primaryLocation)) {
      console.log('✅ Found backend at primary location');
      return primaryLocation;
    }
  }
  
  // List of fallback locations
  const fallbackPaths = [
    path.join(resourcesDir, 'Resources', 'bin', 'main'),
    path.join(resourcesDir, 'app.asar.unpacked', 'bin', 'main'),
    path.join(resourcesDir, '..', 'bin', 'main'),
    // Path with current execPath directory (absolute)
    path.join(path.dirname(process.execPath || ''), '..', 'Resources', 'bin', 'main')
  ];
  
  // Check each possible path
  for (const possiblePath of fallbackPaths) {
    console.log('Checking fallback path:', possiblePath);
    if (fs.existsSync(possiblePath)) {
      console.log('✅ Found backend at fallback path:', possiblePath);
      return possiblePath;
    }
  }
  
  // Last resort - search common paths
  console.log('Searching standard macOS application paths...');
  const standardMacPaths = [
    '/Applications/PocketShield.app/Contents/Resources/bin/main',
    '/Applications/PocketShield.app/Contents/MacOS/bin/main',
    path.join(process.env.HOME || '', 'Applications/PocketShield.app/Contents/Resources/bin/main')
  ];
  
  for (const stdPath of standardMacPaths) {
    console.log('Checking standard path:', stdPath);
    if (fs.existsSync(stdPath)) {
      console.log('✅ Found backend at standard path:', stdPath);
      return stdPath;
    }
  }
  
  // Last resort - try to find and list the resources directory contents
  try {
    console.log('Contents of resources directory:');
    if (fs.existsSync(resourcesDir)) {
      const contents = fs.readdirSync(resourcesDir);
      console.log(contents);
      
      // Check if bin directory exists
      const binDir = path.join(resourcesDir, 'bin');
      if (fs.existsSync(binDir)) {
        console.log('Contents of bin directory:');
        console.log(fs.readdirSync(binDir));
      }
    }
  } catch (error) {
    console.error('Error listing directory contents:', error.message);
  }
  
  console.error('❌ Could not find backend executable in any location');
  return null;
}

async function startBackend() {
  const availablePort = await portfinder.getPortPromise();

  console.log(`Found port available: ${availablePort}`)

  const backendPath = getBackendPath();
  
  if (!backendPath) {
    console.error('🔴 Could not find backend executable');
    return null;
  }
  
  console.log(`Starting backend from: ${backendPath}`);
  
  // Make sure the file is executable (mainly for development)
  try {
    fs.chmodSync(backendPath, '755');
  } catch (error) {
    console.warn('Could not set executable permissions:', error);
  }
  
  // Get the directory of the backend executable to set as cwd
  const backendDir = path.dirname(backendPath);
  
  // Start the Go backend process
  const backend = spawn(backendPath, [], {
    stdio: 'inherit', // This will pipe stdout/stderr to the parent process
    cwd: backendDir, // Set working directory to where the binary is
    env: {
      ...process.env,
      PORT: availablePort.toString(),
      DEBUG: process.env.DEBUG || '*',
      // Add any environment variables needed by the backend
    }
  });
  
  // Handle backend process events
  backend.on('error', (err) => {
    console.error('Failed to start backend:', err);
  });
  
  backend.on('close', (code) => {
    console.log(`Backend process exited with code ${code}`);
    // Only exit this process if backend crashes unexpectedly
    if (code !== 0 && code !== null) {
      process.exit(code);
    }
  });
  
  // Handle termination signals
  process.on('SIGINT', () => {
    console.log('Stopping backend process...');
    backend.kill('SIGINT');
  });
  
  process.on('SIGTERM', () => {
    console.log('Stopping backend process...');
    backend.kill('SIGTERM');
  });
  
  return backend;
}

// Start the backend if this script is called directly
if (require.main === module) {
  startBackend();
}

module.exports = { startBackend }; 