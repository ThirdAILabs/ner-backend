import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// Get __dirname equivalent in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Paths
const projectRoot = path.join(__dirname, '..');
const targetLicenseDir = path.join(projectRoot, 'licenses');

const licenseMetadata = [
  {name: 'ELECTRON', dir: './'},
  {name: 'FRONTEND', dir: '../../frontend'},
  {name: 'PYTHON_PLUGIN', dir: '../../plugin/plugin-python'},
  {name: 'BOLT', dir: '../../internal/core/bolt/lib'},
]

// Ensure license directory exists
if (!fs.existsSync(targetLicenseDir)) {
  console.log('Creating license directory...');
  fs.mkdirSync(targetLicenseDir, { recursive: true });
}

try {
  // Copy licenses
  for (const license of licenseMetadata) {
    const licensePath = path.join(projectRoot, license.dir, 'THIRD_PARTY_NOTICES.txt');
    if (!fs.existsSync(licensePath)) {
      console.error(`${license.name} license not found at: ${licensePath}`);
      process.exit(1);
    }

    const targetLicensePath = path.join(targetLicenseDir, `${license.name}_THIRD_PARTY_NOTICES.txt`);
    fs.cpSync(licensePath, targetLicensePath, {
      recursive: true,
      force: true,
      dereference: false,
      preserveTimestamps: true
    });
  }
  console.log('Licenses copied successfully!');
} catch (error) {
  console.error('Failed to copy licenses:', error.message);
  process.exit(1);
}