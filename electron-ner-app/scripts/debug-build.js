const { execSync, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Project paths
const rootDir = path.join(__dirname, '..');
const goProjectDir = path.join(rootDir, '..');
const binDir = path.join(rootDir, 'bin');
const appDistDir = path.join(rootDir, 'dist');
const appMacDir = path.join(appDistDir, 'mac');

// Command line arguments
const args = process.argv.slice(2);
const command = args[0];

// Helper functions
function ensureBinDirectory() {
  if (!fs.existsSync(binDir)) {
    console.log('Creating bin directory...');
    fs.mkdirSync(binDir, { recursive: true });
  }
}

function copyBackend() {
  const backendExecutable = path.join(goProjectDir, 'main');
  const targetExecutable = path.join(binDir, 'main');
  
  if (!fs.existsSync(backendExecutable)) {
    console.error('Error: Backend executable not found at:', backendExecutable);
    process.exit(1);
  }
  
  console.log(`Copying backend from ${backendExecutable} to ${targetExecutable}`);
  fs.copyFileSync(backendExecutable, targetExecutable);
  fs.chmodSync(targetExecutable, '755');
  console.log('Backend copied successfully!');
  return targetExecutable;
}

// Execute the specified test
switch (command) {
  case 'check-backend':
    // Just check if backend is available
    const backendExecutable = path.join(goProjectDir, 'main');
    if (fs.existsSync(backendExecutable)) {
      console.log('✅ Backend executable found at:', backendExecutable);
      console.log('File size:', fs.statSync(backendExecutable).size, 'bytes');
    } else {
      console.error('❌ Backend executable not found at:', backendExecutable);
      process.exit(1);
    }
    break;
    
  case 'build-frontend-only':
    // Only build the frontend with Vite
    console.log('Building frontend only...');
    execSync('npm run vite-build', {
      cwd: rootDir,
      stdio: 'inherit'
    });
    console.log('✅ Frontend built successfully!');
    break;
    
  case 'quick-package':
    // Quickly build a development package for testing (no signing, no DMG)
    console.log('Building quick test package...');
    execSync('npm run vite-build', {
      cwd: rootDir,
      stdio: 'inherit'
    });
    
    ensureBinDirectory();
    copyBackend();
    
    console.log('Creating quick test package...');
    execSync('electron-builder --dir --x64', {
      cwd: rootDir,
      stdio: 'inherit'
    });
    
    console.log('✅ Quick package built in:', appMacDir);
    break;
    
  case 'test-backend-in-package':
    // Test running the backend from the package
    if (!fs.existsSync(appMacDir)) {
      console.error('❌ App package not found. Run quick-package first.');
      process.exit(1);
    }
    
    const packagedBackend = path.join(appMacDir, 'NER Electron App.app', 'Contents', 'Resources', 'bin', 'main');
    
    if (!fs.existsSync(packagedBackend)) {
      console.error('❌ Backend not found in package:', packagedBackend);
      process.exit(1);
    }
    
    console.log('✅ Backend found in package:', packagedBackend);
    console.log('Running packaged backend executable directly...');
    
    // Make it executable
    fs.chmodSync(packagedBackend, '755');
    
    // Run the backend executable directly
    const backend = spawn(packagedBackend, [], {
      stdio: 'inherit',
    });
    
    backend.on('error', (err) => {
      console.error('Failed to start backend:', err);
    });
    
    // Handle Ctrl+C to terminate the process
    process.on('SIGINT', () => {
      console.log('Stopping backend process...');
      backend.kill('SIGINT');
      process.exit(0);
    });
    break;
    
  case 'run-app-with-logging':
    // Run the packaged app with console logging
    if (!fs.existsSync(appMacDir)) {
      console.error('❌ App package not found. Run quick-package first.');
      process.exit(1);
    }
    
    const appPath = path.join(appMacDir, 'NER Electron App.app', 'Contents', 'MacOS', 'NER Electron App');
    
    if (!fs.existsSync(appPath)) {
      console.error('❌ App executable not found:', appPath);
      process.exit(1);
    }
    
    console.log('✅ App executable found:', appPath);
    console.log('Running packaged app with logging...');
    
    // Run the app with DEBUG environment variables
    const app = spawn(appPath, [], {
      stdio: 'inherit',
      env: {
        ...process.env,
        ELECTRON_ENABLE_LOGGING: 1,
        ELECTRON_ENABLE_STACK_DUMPING: 1,
        DEBUG: '*'
      }
    });
    
    app.on('error', (err) => {
      console.error('Failed to start app:', err);
    });
    
    // Handle Ctrl+C to terminate the process
    process.on('SIGINT', () => {
      console.log('Stopping app process...');
      app.kill('SIGINT');
      process.exit(0);
    });
    break;
    
  default:
    console.log(`
Debug Build Commands:
  check-backend         - Check if the backend executable exists
  build-frontend-only   - Build only the frontend with Vite
  quick-package         - Create a quick test package without DMG
  test-backend-in-package - Run the backend directly from the package
  run-app-with-logging  - Run the packaged app with console logging
    `);
} 