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
const modelPath = process.env.MODEL_PATH;
if (!modelPath) {
  console.error('MODEL_PATH environment variable is not set');
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

// Handle both file and directory paths
const stats = fs.statSync(modelPath);
if (stats.isDirectory()) {
  // Look for .model files in the directory
  const files = fs.readdirSync(modelPath);
  const modelFiles = files.filter(file => file.endsWith('.model'));
  
  if (modelFiles.length === 0) {
    console.error('No .model files found in directory:', modelPath);
    process.exit(1);
  }

  // Copy all model files
  for (const modelFile of modelFiles) {
    const sourceFile = path.join(modelPath, modelFile);
    const targetFile = path.join(binDir, modelFile);
    
    try {
      console.log(`Copying model from ${sourceFile} to ${targetFile}`);
      fs.copyFileSync(sourceFile, targetFile);
      console.log(`Model ${modelFile} copied successfully!`);
    } catch (error) {
      console.error(`Failed to copy model ${modelFile}:`, error.message);
      process.exit(1);
    }
  }
} else {
  // Handle single file
  const modelFileName = path.basename(modelPath);
  const targetModelPath = path.join(binDir, modelFileName);

  try {
    console.log(`Copying model from ${modelPath} to ${targetModelPath}`);
    fs.copyFileSync(modelPath, targetModelPath);
    console.log('Model copied successfully!');
  } catch (error) {
    console.error('Failed to copy model:', error.message);
    process.exit(1);
  }
}
