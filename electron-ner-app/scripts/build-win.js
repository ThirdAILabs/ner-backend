import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { exec } from 'node:child_process';
import { promisify } from 'node:util';

const execAsync = promisify(exec);

// Get __dirname equivalent in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Project paths
const rootDir = path.join(__dirname, '..');
const goProjectDir = path.join(rootDir, '..');

console.log('Starting full build process for Windows...');
console.log('Root directory:', rootDir);
console.log('Go project directory:', goProjectDir);

// Step 1: Check/Build the Go backend executable for Windows
console.log('\n=== Checking Go backend ===');
try {
  const backendExecutable = path.join(goProjectDir, 'main.exe');
  const backendLinuxExecutable = path.join(goProjectDir, 'main');
  
  if (!fs.existsSync(backendExecutable)) {
    console.log('Windows backend executable not found, building it...');
    try {
      // Try to build the Windows executable
      execSync('set GOOS=windows && set GOARCH=amd64 && go build -o main.exe', {
        cwd: goProjectDir,
        stdio: 'inherit',
        shell: true
      });
    } catch (error) {
      console.error('Failed to build Windows executable:', error.message);
      
      // Fall back to checking for Linux/Mac executable
      if (fs.existsSync(backendLinuxExecutable)) {
        console.warn('Using non-Windows backend executable. This may not work in the final package on Windows.');
        
        // Copy the non-Windows executable as main.exe
        fs.copyFileSync(backendLinuxExecutable, backendExecutable);
      } else {
        console.error('No backend executable found. Please build the backend first with:');
        console.error('  go build -o main.exe      # on Windows');
        console.error('  GOOS=windows GOARCH=amd64 go build -o main.exe   # on other platforms');
        process.exit(1);
      }
    }
  }
  
  console.log('Go backend executable found or created at:', backendExecutable);
} catch (error) {
  console.error('Failed to check/build Go backend:', error.message);
  process.exit(1);
}

// Step 2: Copy the backend to bin directory
console.log('\n=== Copying backend to electron app ===');
try {
  const backendExecutable = path.join(goProjectDir, 'main.exe');
  const binDir = path.join(rootDir, 'bin');
  const targetExecutable = path.join(binDir, 'main.exe');
  
  // Ensure bin directory exists
  if (!fs.existsSync(binDir)) {
    fs.mkdirSync(binDir, { recursive: true });
  }
  
  // Copy the file
  console.log(`Copying from ${backendExecutable} to ${targetExecutable}`);
  fs.copyFileSync(backendExecutable, targetExecutable);
  
  console.log('Backend copied successfully!');
} catch (error) {
  console.error('Failed to copy backend:', error.message);
  process.exit(1);
}

// Step 3: Build the frontend with Vite
console.log('\n=== Building frontend with Vite ===');
try {
  execSync('npm run vite-build', {
    cwd: rootDir,
    stdio: 'inherit'
  });
  
  console.log('Frontend built successfully!');
} catch (error) {
  console.error('Failed to build frontend:', error.message);
  process.exit(1);
}

// Step 4: Package with electron-builder
console.log('\n=== Packaging with electron-builder ===');
try {
  // Make sure we use NODE_ENV=production for packaging
  const command = process.platform === 'win32' 
    ? 'set NODE_ENV=production && npm run electron-build -- --win --x64' 
    : 'NODE_ENV=production npm run electron-build -- --win --x64';
  
  execSync(command, {
    cwd: rootDir,
    stdio: 'inherit',
    shell: true
  });
  
  console.log('Windows package built successfully!');
} catch (error) {
  console.error('Failed to build Windows package:', error.message);
  process.exit(1);
}

console.log('\n=== Build complete! ===');
console.log('Your Windows package is located in the dist directory.');
console.log('The backend is automatically packaged with the application.'); 