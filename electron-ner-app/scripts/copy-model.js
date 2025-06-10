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

// Check if model file exists
if (!fs.existsSync(modelPath)) {
  console.error('Model file not found at:', modelPath);
  process.exit(1);
}

// Copy all files from the model folder to the bin directory
const modelFolderName = path.basename(modelPath);
const targetModelFolderPath = path.join(binDir, modelFolderName);

try {
  console.log(`Copying model folder from ${modelPath} to ${targetModelFolderPath}`);
  
  // Ensure target folder exists
  if (!fs.existsSync(targetModelFolderPath)) {
    fs.mkdirSync(targetModelFolderPath, { recursive: true });
  }

  // Read all files in the model folder
  const files = fs.readdirSync(modelPath);
  files.forEach(file => {
    const sourceFilePath = path.join(modelPath, file);
    const targetFilePath = path.join(targetModelFolderPath, file);

    // Copy each file
    fs.copyFileSync(sourceFilePath, targetFilePath);
    console.log(`Copied ${file} to ${targetFilePath}`);
  });

  console.log('Model folder copied successfully!');
} catch (error) {
  console.error('Failed to copy model files:', error.message);
  process.exit(1);
}
