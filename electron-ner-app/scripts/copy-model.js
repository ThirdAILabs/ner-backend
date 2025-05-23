import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// Get __dirname equivalent in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Paths
const projectRoot = path.join(__dirname, '..');
const binDir = path.join(projectRoot, 'bin');

// Get model path from environment variable
const modelPath = process.env.MODEL_DIR;
if (!modelPath) {
  console.error('MODEL_DIR environment variable is not set');
  process.exit(1);
}

const modelType = process.env.MODEL_TYPE;
if (!modelType) {
  console.error('MODEL_TYPE environment variable is not set');
  process.exit(1);
}

// Ensure bin directory exists
if (!fs.existsSync(binDir)) {
  console.log('Creating bin directory...');
  fs.mkdirSync(binDir, { recursive: true });
}

// Check if model path exists
if (!fs.existsSync(modelPath)) {
  console.error('Model path not found at:', modelPath);
  process.exit(1);
}

// Copy only the specific model type directory
try {
  const modelTypeDir = path.join(modelPath, modelType);
  const targetModelDir = path.join(binDir, modelType);

  if (!fs.existsSync(modelTypeDir)) {
    console.error(`Model type directory not found at: ${modelTypeDir}`);
    process.exit(1);
  }

  console.log(`Copying ${modelType} model from ${modelTypeDir} to ${targetModelDir}`);
  fs.cpSync(modelTypeDir, targetModelDir, { 
    recursive: true, 
    force: true,
    dereference: false,
    preserveTimestamps: true
  });
  console.log(`${modelType} model directory copied successfully!`);
} catch (error) {
  console.error('Failed to copy model files:', error.message);
  process.exit(1);
}
