const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

// Project paths
const rootDir = path.join(__dirname, '..');
const goProjectDir = path.join(rootDir, '..');

console.log('Starting full build process for macOS DMG...');
console.log('Root directory:', rootDir);
console.log('Go project directory:', goProjectDir);

// Step 1: Check the Go backend executable
console.log('\n=== Checking Go backend ===');
try {
  const backendExecutable = path.join(goProjectDir, 'main');
  
  if (!fs.existsSync(backendExecutable)) {
    console.error('Error: Go backend executable not found at:', backendExecutable);
    console.error('Please make sure you have built the backend first with "go build -o main"');
    process.exit(1);
  }
  
  console.log('Go backend executable found at:', backendExecutable);
} catch (error) {
  console.error('Failed to check Go backend:', error.message);
  process.exit(1);
}

// Step 2: Copy the backend to bin directory
console.log('\n=== Copying backend to electron app ===');
try {
  const backendExecutable = path.join(goProjectDir, 'main');
  const binDir = path.join(rootDir, 'bin');
  const targetExecutable = path.join(binDir, 'main');
  
  // Ensure bin directory exists
  if (!fs.existsSync(binDir)) {
    fs.mkdirSync(binDir, { recursive: true });
  }
  
  // Copy the file
  console.log(`Copying from ${backendExecutable} to ${targetExecutable}`);
  fs.copyFileSync(backendExecutable, targetExecutable);
  fs.chmodSync(targetExecutable, '755');
  
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
  execSync('NODE_ENV=production npm run electron-build -- --mac --x64', {
    cwd: rootDir,
    stdio: 'inherit'
  });
  
  console.log('DMG package built successfully!');
} catch (error) {
  console.error('Failed to build DMG package:', error.message);
  process.exit(1);
}

console.log('\n=== Build complete! ===');
console.log('Your DMG package is located in the dist directory.');
console.log('The backend is automatically packaged with the application.'); 