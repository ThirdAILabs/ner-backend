import fs from 'node:fs';
import path from 'node:path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'node:url';

// Get __dirname equivalent in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Paths
const projectRoot = path.join(__dirname, '..');
const binDir = path.join(projectRoot, 'bin');
const goProjectDir = path.join(projectRoot, '..');
const backendExecutable = path.join(goProjectDir, 'main');
const targetExecutable = path.join(binDir, 'main');

// Ensure bin directory exists
if (!fs.existsSync(binDir)) {
  console.log('Creating bin directory...');
  fs.mkdirSync(binDir, { recursive: true });
}

// Check if Go executable exists
if (!fs.existsSync(backendExecutable)) {
  console.error('Backend executable not found at:', backendExecutable);
  console.log('Trying to build Go backend...');
  
  try {
    // Try to build the backend
    execSync('go build -o main', { cwd: goProjectDir, stdio: 'inherit' });
    console.log('Go backend built successfully.');
  } catch (error) {
    console.error('Failed to build Go backend:', error.message);
    process.exit(1);
  }
}

// Copy the backend executable to bin directory
try {
  console.log(`Copying backend from ${backendExecutable} to ${targetExecutable}`);
  fs.copyFileSync(backendExecutable, targetExecutable);
  
  // Make it executable
  fs.chmodSync(targetExecutable, '755');
  
  console.log('Backend copied successfully!');
} catch (error) {
  console.error('Failed to copy backend:', error.message);
  process.exit(1);
} 