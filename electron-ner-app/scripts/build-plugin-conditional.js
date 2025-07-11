import fs from 'node:fs';
import path from 'node:path';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const BUILD_PLUGIN = process.env.BUILD_PLUGIN === 'TRUE';
const MODEL_TYPE = process.env.MODEL_TYPE || '';
const ENABLE_PYTHON = process.env.ENABLE_PYTHON === 'TRUE' || process.env.ENABLE_PYTHON === 'true';

// Check if plugin dist directory exists and is not empty
const pluginDistPath = path.join(__dirname, '../../plugin/plugin-python/dist');
let pluginDistEmpty = true;
if (fs.existsSync(pluginDistPath)) {
  const files = fs.readdirSync(pluginDistPath);
  pluginDistEmpty = files.length === 0;
}

// Check conditions
const shouldBuildPlugin = (
  (BUILD_PLUGIN || !fs.existsSync(pluginDistPath) || pluginDistEmpty) &&
  (MODEL_TYPE.startsWith('python_') || MODEL_TYPE.startsWith('onnx_')) &&
  ENABLE_PYTHON
);

if (shouldBuildPlugin) {
  console.log('Building plugin...');
  try {
    execSync('npm run build-plugin', { stdio: 'inherit' });
  } catch (error) {
    console.error('Failed to build plugin:', error.message);
    process.exit(1);
  }
} else {
  console.log('Skipping plugin build');
}