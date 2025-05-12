#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');
const { program } = require('commander');

// Project paths
const rootDir = path.join(__dirname, '..');
const binDir = path.join(rootDir, 'bin');
const appDir = path.join(rootDir, 'dist', 'mac', 'PocketShield.app');
const resourcesDir = path.join(appDir, 'Contents', 'Resources');
const macosDir = path.join(appDir, 'Contents', 'MacOS');
const appExecutable = path.join(macosDir, 'PocketShield');

// Define program and commands
program
  .name('debug-packaging')
  .description('Debug tools for the electron-ner-app packaging')
  .version('1.0.0');

// Command to inspect package structure
program
  .command('inspect')
  .description('Inspect the packaged app structure')
  .action(() => {
    console.log('Inspecting packaged app structure...');
    
    if (!fs.existsSync(appDir)) {
      console.error('❌ App package not found. Run quick-package first.');
      return;
    }
    
    // Check app executable
    console.log('\n📦 Executable:');
    if (fs.existsSync(appExecutable)) {
      console.log('✅ App executable found:', appExecutable);
      console.log('File size:', fs.statSync(appExecutable).size, 'bytes');
      // Make it executable
      try {
        fs.chmodSync(appExecutable, '755');
        console.log('Made executable (chmod 755)');
      } catch (error) {
        console.warn('Could not set permissions:', error.message);
      }
    } else {
      console.error('❌ App executable not found');
    }
    
    // Check resources
    console.log('\n📦 Resources:');
    if (fs.existsSync(resourcesDir)) {
      console.log('✅ Resources directory found');
      // List resources
      try {
        const resources = fs.readdirSync(resourcesDir);
        console.log('Resources:', resources);
        
        // Check for app.asar
        const asarPath = path.join(resourcesDir, 'app.asar');
        if (fs.existsSync(asarPath)) {
          console.log('✅ app.asar found, size:', fs.statSync(asarPath).size, 'bytes');
        } else {
          console.error('❌ app.asar not found');
        }
        
        // Check unpacked resources
        const unpackedPath = path.join(resourcesDir, 'app.asar.unpacked');
        if (fs.existsSync(unpackedPath)) {
          console.log('✅ app.asar.unpacked found');
          try {
            const unpacked = fs.readdirSync(unpackedPath);
            console.log('Unpacked resources:', unpacked);
          } catch (error) {
            console.error('Error reading unpacked directory:', error.message);
          }
        } else {
          console.log('ℹ️ No app.asar.unpacked found (this is okay if nothing is unpacked)');
        }
      } catch (error) {
        console.error('Error reading resources:', error.message);
      }
    } else {
      console.error('❌ Resources directory not found');
    }
    
    // Check bin directory
    console.log('\n📦 Backend Binary:');
    const binPaths = [
      path.join(resourcesDir, 'bin', 'main'),
      path.join(resourcesDir, 'bin'),
      path.join(resourcesDir, 'app.asar.unpacked', 'bin', 'main'),
      path.join(resourcesDir, 'Resources', 'bin', 'main'),
    ];
    
    let found = false;
    for (const binPath of binPaths) {
      if (fs.existsSync(binPath)) {
        if (fs.statSync(binPath).isFile()) {
          console.log('✅ Backend binary found at:', binPath);
          console.log('File size:', fs.statSync(binPath).size, 'bytes');
          found = true;
          // Make it executable
          try {
            fs.chmodSync(binPath, '755');
            console.log('Made executable (chmod 755)');
          } catch (error) {
            console.warn('Could not set permissions:', error.message);
          }
        } else if (fs.statSync(binPath).isDirectory()) {
          console.log('✅ Backend bin directory found at:', binPath);
          try {
            const binContents = fs.readdirSync(binPath);
            console.log('Contents:', binContents);
          } catch (error) {
            console.error('Error reading bin directory:', error.message);
          }
        }
      }
    }
    
    if (!found) {
      console.error('❌ Backend binary not found in any expected location');
    }
  });

// Command to copy the backend to the right location
program
  .command('fix-backend')
  .description('Fix the backend binary location in the packaged app')
  .action(() => {
    console.log('Fixing backend binary location...');
    
    if (!fs.existsSync(appDir)) {
      console.error('❌ App package not found. Run quick-package first.');
      return;
    }
    
    // Source binary
    const sourceBinary = path.join(rootDir, '..', 'main');
    if (!fs.existsSync(sourceBinary)) {
      console.error('❌ Source binary not found at:', sourceBinary);
      return;
    }
    
    // Target directories
    const binDirectories = [
      path.join(resourcesDir, 'bin'),
      path.join(resourcesDir, 'Resources', 'bin'),
    ];
    
    for (const binDir of binDirectories) {
      try {
        // Ensure directory exists
        if (!fs.existsSync(binDir)) {
          console.log('Creating directory:', binDir);
          fs.mkdirSync(binDir, { recursive: true });
        }
        
        // Copy the binary
        const targetBinary = path.join(binDir, 'main');
        console.log(`Copying from ${sourceBinary} to ${targetBinary}`);
        fs.copyFileSync(sourceBinary, targetBinary);
        fs.chmodSync(targetBinary, '755');
        console.log('✅ Binary copied successfully to:', targetBinary);
      } catch (error) {
        console.error(`Error with directory ${binDir}:`, error.message);
      }
    }
  });

// Command to run just the backend from the packaged app
program
  .command('run-backend')
  .description('Run the backend from the packaged app')
  .action(() => {
    console.log('Running backend from packaged app...');
    
    if (!fs.existsSync(appDir)) {
      console.error('❌ App package not found. Run quick-package first.');
      return;
    }
    
    // Find backend binary
    const binPaths = [
      path.join(resourcesDir, 'bin', 'main'),
      path.join(resourcesDir, 'Resources', 'bin', 'main'),
      path.join(resourcesDir, 'app.asar.unpacked', 'bin', 'main'),
    ];
    
    let backendPath = null;
    for (const binPath of binPaths) {
      if (fs.existsSync(binPath)) {
        backendPath = binPath;
        break;
      }
    }
    
    if (!backendPath) {
      console.error('❌ Backend binary not found. Run fix-backend first.');
      return;
    }
    
    console.log('Starting backend from:', backendPath);
    
    // Make executable
    fs.chmodSync(backendPath, '755');
    
    // Run the backend
    const backend = spawn(backendPath, [], {
      stdio: 'inherit',
    });
    
    backend.on('error', (err) => {
      console.error('Failed to start backend:', err.message);
    });
    
    // Handle Ctrl+C
    process.on('SIGINT', () => {
      console.log('Stopping backend...');
      backend.kill('SIGINT');
      process.exit(0);
    });
  });

// Command to run the app with extra debug info
program
  .command('run-app')
  .description('Run the packaged app with debug environment')
  .action(() => {
    console.log('Running packaged app with debug settings...');
    
    if (!fs.existsSync(appExecutable)) {
      console.error('❌ App executable not found. Run quick-package first.');
      return;
    }
    
    // First, fix the backend location to be sure
    try {
      program.executeSubCommand(['fix-backend']);
    } catch (error) {
      console.warn('Warning during fix-backend:', error.message);
    }
    
    console.log('Starting app from:', appExecutable);
    
    // Make executable
    fs.chmodSync(appExecutable, '755');
    
    // Run with debug environment
    const app = spawn(appExecutable, [], {
      stdio: 'inherit',
      env: {
        ...process.env,
        ELECTRON_ENABLE_LOGGING: '1',
        ELECTRON_ENABLE_STACK_DUMPING: '1',
        NODE_ENV: 'production',
        DEBUG: '*',
      },
    });
    
    app.on('error', (err) => {
      console.error('Failed to start app:', err.message);
    });
    
    // Handle Ctrl+C
    process.on('SIGINT', () => {
      console.log('Stopping app...');
      app.kill('SIGINT');
      process.exit(0);
    });
  });

// Command to create a quick package for testing
program
  .command('quick-package')
  .description('Create a quick package for testing')
  .action(() => {
    console.log('Creating quick package...');
    
    // Build frontend
    console.log('\nBuilding frontend...');
    execSync('npm run vite-build', {
      cwd: rootDir,
      stdio: 'inherit',
    });
    
    // Ensure bin directory exists
    if (!fs.existsSync(binDir)) {
      fs.mkdirSync(binDir, { recursive: true });
    }
    
    // Copy backend
    const sourceBinary = path.join(rootDir, '..', 'main');
    const targetBinary = path.join(binDir, 'main');
    
    if (fs.existsSync(sourceBinary)) {
      console.log(`\nCopying backend from ${sourceBinary} to ${targetBinary}`);
      fs.copyFileSync(sourceBinary, targetBinary);
      fs.chmodSync(targetBinary, '755');
    } else {
      console.error('❌ Source binary not found at:', sourceBinary);
      return;
    }
    
    // Package with electron-builder
    console.log('\nPackaging with electron-builder...');
    execSync('electron-builder --dir --x64', {
      cwd: rootDir,
      stdio: 'inherit',
    });
    
    console.log('\n✅ Quick package created successfully!');
  });

// Parse command line arguments
program.parse();

// Show help if no command specified
if (!process.argv.slice(2).length) {
  program.outputHelp();
} 