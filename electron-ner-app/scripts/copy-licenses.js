import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// Get __dirname equivalent in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Paths
const projectRoot = path.join(__dirname, '..');
const projectLicense = path.join(projectRoot, 'THIRD_PARTY_NOTICES.txt');
const frontendRoot = path.join(__dirname, '../../frontend');
const frontendLicense = path.join(frontendRoot, 'THIRD_PARTY_NOTICES.txt');
const licenseDir = path.join(projectRoot, 'licenses');

// Ensure license directory exists
if (!fs.existsSync(licenseDir)) {
  console.log('Creating license directory...');
  fs.mkdirSync(licenseDir, { recursive: true });
}

// Copy project license
if (!fs.existsSync(projectLicense)) {
  console.error('Project license not found at:', projectLicense);
  process.exit(1);
}

// Copy frontend license
if (!fs.existsSync(frontendLicense)) {
  console.error('Frontend license not found at:', frontendLicense);
  process.exit(1);
}

// Copy project license
try {
  const targetProjectLicense = path.join(licenseDir, 'THIRD_PARTY_NOTICES.txt');

  fs.cpSync(projectLicense, targetProjectLicense, {
    recursive: true,
    force: true,
    dereference: false,
    preserveTimestamps: true
  });

  // Read frontend license content
  const frontendLicenseContent = fs.readFileSync(frontendLicense, 'utf8');
  
  // Replace first two lines with dashes
  const modifiedContent = frontendLicenseContent.split('\n')
    .slice(2)
    .join('\n');
  
  // Append modified content to project license
  fs.appendFileSync(targetProjectLicense, '\n\n-----------\n' + modifiedContent);

  console.log('Licenses copied successfully!');
} catch (error) {
  console.error('Failed to copy licenses:', error.message);
  process.exit(1);
}