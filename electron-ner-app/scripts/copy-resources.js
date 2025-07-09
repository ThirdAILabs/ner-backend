import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'module';

// Create require for CommonJS modules
const require = createRequire(import.meta.url);
const { isMacOS } = require('./platform-utils.cjs');

// Get __dirname equivalent in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Paths
const projectRoot = path.join(__dirname, '..');
const resourcesDir = path.join(projectRoot, 'resources');

// Ensure resources directory exists
if (!fs.existsSync(resourcesDir)) {
  console.log('Creating resources directory...');
  fs.mkdirSync(resourcesDir, { recursive: true });
}

// Copy macOS dylib files if on macOS and they exist
if (isMacOS()) {
  const buildDir = path.join(projectRoot, 'build');
  const distFrameworksDir = path.join(projectRoot, 'dist', 'mac-arm64', 'PocketShield.app', 'Contents', 'Frameworks');
  
  const dylibFiles = ['libonnxruntime.dylib', 'libomp.dylib'];
  
  for (const dylib of dylibFiles) {
    let copied = false;
    
    // Try to copy from build directory first
    const buildPath = path.join(buildDir, dylib);
    if (fs.existsSync(buildPath)) {
      const targetPath = path.join(resourcesDir, dylib);
      console.log(`Copying ${dylib} from build directory...`);
      fs.copyFileSync(buildPath, targetPath);
      copied = true;
    }
    
    // Try dist directory if not found in build
    if (!copied) {
      const distPath = path.join(distFrameworksDir, dylib);
      if (fs.existsSync(distPath)) {
        const targetPath = path.join(resourcesDir, dylib);
        console.log(`Copying ${dylib} from dist directory...`);
        fs.copyFileSync(distPath, targetPath);
        copied = true;
      }
    }
    
    if (!copied) {
      console.warn(`Warning: ${dylib} not found in build or dist directories`);
    }
  }
  
  console.log('macOS resources copied successfully!');
}