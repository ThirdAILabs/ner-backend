#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Paths
const rootDir = path.join(__dirname, '..');
const distDir = path.join(rootDir, 'dist');
const sourceBackend = path.join(rootDir, '..', 'main');
const dmgFile = path.join(distDir, 'PocketShield-1.0.0.dmg');
const macAppDir = path.join(distDir, 'mac', 'PocketShield.app');
const targetBinDir = path.join(macAppDir, 'Contents', 'Resources', 'bin');

console.log('Fixing backend binary in DMG package...');

// Check if source binary exists
if (!fs.existsSync(sourceBackend)) {
  console.error('❌ Source backend binary not found at:', sourceBackend);
  process.exit(1);
}

// Check if the .app directory exists (dir build)
if (fs.existsSync(macAppDir)) {
  console.log('Found .app directory, copying backend...');
  
  // Make sure bin directory exists
  if (!fs.existsSync(targetBinDir)) {
    console.log('Creating bin directory in app bundle...');
    fs.mkdirSync(targetBinDir, { recursive: true });
  }
  
  // Copy the binary
  const targetBackend = path.join(targetBinDir, 'main');
  console.log(`Copying from ${sourceBackend} to ${targetBackend}`);
  fs.copyFileSync(sourceBackend, targetBackend);
  fs.chmodSync(targetBackend, '755');
  console.log('✅ Backend copied to app bundle successfully!');
}

// Check if DMG exists
if (fs.existsSync(dmgFile)) {
  console.log('Found DMG file, but we cannot modify it directly.');
  console.log('Please install the app and run the following if needed:');
  console.log(`sudo mkdir -p "/Applications/PocketShield.app/Contents/Resources/bin"`);
  console.log(`sudo cp "${sourceBackend}" "/Applications/PocketShield.app/Contents/Resources/bin/main"`);
  console.log(`sudo chmod 755 "/Applications/PocketShield.app/Contents/Resources/bin/main"`);
} else {
  console.log('⚠️ DMG file not found. Only directory package was updated.');
}

// Generate an installation script that can be run after installing the app
const installScript = path.join(distDir, 'install-backend.sh');
const scriptContent = `#!/bin/bash
# Script to install backend binary into the application
echo "Installing backend binary into the application..."
sudo mkdir -p "/Applications/PocketShield.app/Contents/Resources/bin"
sudo cp "${sourceBackend}" "/Applications/PocketShield.app/Contents/Resources/bin/main"
sudo chmod 755 "/Applications/PocketShield.app/Contents/Resources/bin/main"
echo "Backend installed successfully!"
`;

// Write the script
fs.writeFileSync(installScript, scriptContent);
fs.chmodSync(installScript, '755');
console.log(`✅ Created installation script at: ${installScript}`);
console.log('You can run this script after installing the app to ensure backend is available.');

console.log('Done!'); 