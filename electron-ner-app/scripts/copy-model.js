const fs = require('fs');
const path = require('path');

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

// Check if model file exists
if (!fs.existsSync(modelPath)) {
  console.error('Model file not found at:', modelPath);
  process.exit(1);
}

// Copy the model file to bin directory
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
