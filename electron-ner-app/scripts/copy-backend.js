import fs from 'node:fs';
import path from 'node:path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'module';

// Create require for CommonJS modules
const require = createRequire(import.meta.url);
const { isWindows, getExecutableName, getWindowsDependencies } = require('./platform-utils.cjs');

// Get __dirname equivalent in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Paths
const projectRoot = path.join(__dirname, '..');
const binDir = path.join(projectRoot, 'bin');
const goProjectDir = path.join(projectRoot, '..');
const backendExecutable = path.join(goProjectDir, getExecutableName('main'));
const targetExecutable = path.join(binDir, getExecutableName('main'));

// Plugin paths
const pluginDir = path.join(projectRoot, '..', 'plugin/plugin-python/dist/plugin');
const targetPluginDir = path.join(binDir, 'plugin');

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
    const outputName = getExecutableName('main');
    execSync(`go build -o ${outputName}`, { cwd: goProjectDir, stdio: 'inherit' });
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
  
  // Copy Windows dependencies if on Windows
  if (isWindows()) {
    const windowsDeps = getWindowsDependencies();
    for (const dep of windowsDeps) {
      const sourcePath = path.join(goProjectDir, dep);
      const targetPath = path.join(binDir, dep);
      if (fs.existsSync(sourcePath)) {
        console.log(`Copying Windows dependency: ${dep}`);
        fs.copyFileSync(sourcePath, targetPath);
      } else {
        console.warn(`Windows dependency not found: ${sourcePath}`);
      }
    }
  }
  
  // Make it executable
  fs.chmodSync(targetExecutable, '755');
  
  console.log('Backend copied successfully!');
} catch (error) {
  console.error('Failed to copy backend:', error.message);
  process.exit(1);
}

function copyRecursivePreservingSymlinks(src, dest) {
  const stats = fs.lstatSync(src); // Use lstat to get info about the link itself, not its target

  if (stats.isDirectory()) {
    fs.mkdirSync(dest, { recursive: true }); // Create destination directory
    const srcDirStat = fs.statSync(src); // Get original directory stats for timestamps

    for (const item of fs.readdirSync(src)) { // Iterate over items in source directory
      copyRecursivePreservingSymlinks(
        path.join(src, item),
        path.join(dest, item) // Copy each item into the new destination directory
      );
    }
    // Set timestamps for the directory after all its children are copied.
    try {
      fs.utimesSync(dest, srcDirStat.atime, srcDirStat.mtime);
    } catch (err) {
      console.warn(`Could not set timestamps for directory ${dest}: ${err.message}`);
    }
  } else if (stats.isSymbolicLink()) {
    const linkTarget = fs.readlinkSync(src); // Read the original link's target path
    fs.symlinkSync(linkTarget, dest); // Create new symlink with the exact same target string

    // Preserve timestamps for the symlink itself, if possible (Node v16+)
    if (fs.lutimesSync) {
        try {
            const srcLinkStat = fs.lstatSync(src); // Get original link's stats
            fs.lutimesSync(dest, srcLinkStat.atime, srcLinkStat.mtime);
        } catch (err) {
            console.warn(`Could not set timestamps for symlink ${dest}: ${err.message}`);
        }
    }
  } else { // It's a regular file
    fs.copyFileSync(src, dest); // Copies data and mode. Preserves timestamps by default.
  }
}

// Copy the plugin directory
if (process.env.MODEL_TYPE?.startsWith('python_') || process.env.MODEL_TYPE?.startsWith('onnx_')) {
  try {
    console.log(`Copying plugin directory from ${pluginDir} to ${targetPluginDir}`);
    fs.rmSync(targetPluginDir, { recursive: true, force: true }); // Remove existing target directory
    fs.mkdirSync(targetPluginDir, { recursive: true }); // Create new target directory

    // Copy entire plugin directory using the custom function
    if (fs.existsSync(pluginDir)) {
      // Iterate over the items in the root of pluginDir and copy them to targetPluginDir
      const items = fs.readdirSync(pluginDir);
      for (const item of items) {
          copyRecursivePreservingSymlinks(
              path.join(pluginDir, item),
              path.join(targetPluginDir, item)
          );
      }
      
      // Make plugin executable (assuming it's named 'plugin' at the top level)
      const pluginExecutableName = 'plugin'; // As in your original script
      const targetPluginExe = path.join(targetPluginDir, pluginExecutableName);
      
      if (fs.existsSync(targetPluginExe)) {
        // fs.chmodSync will operate on the symlink's target on most POSIX systems if targetPluginExe is a symlink.
        fs.chmodSync(targetPluginExe, '755');
        console.log(`Executable permissions set for ${targetPluginExe}`);
      } else {
        // It's possible the executable is not at the top level or has a different name.
        // You might want to adjust this warning or the logic if 'plugin' isn't always the name/location.
        console.warn(`Plugin executable '${pluginExecutableName}' not found at ${targetPluginExe}. Permissions not set.`);
      }
      
      console.log('Plugin directory copied successfully using custom function!');
    } else {
      console.error(`Plugin directory not found at: ${pluginDir}`);
      process.exit(1); // Exit if source plugin directory doesn't exist
    }
  } catch (error) {
    console.error('Failed to copy plugin directory:', error.message);
    console.error(error.stack); // It's good practice to log the stack for better debugging
    process.exit(1); // Exit on any error during the copy process
  }
} else {
  console.log('Skipping plugin directory copy as MODEL_TYPE is not a python model');
}