import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { exec } from 'node:child_process';
import { promisify } from 'node:util';

const execAsync = promisify(exec);

// Get __dirname equivalent in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * This hook runs after Electron Builder packs the app but before it creates the installer
 * This ensures the backend binary is properly copied to the final location in the app bundle
 */
export async function afterPack(context) {
  const { appOutDir, packager, electronPlatformName } = context;
  const appDir = packager.info._appDir;
  const sourceBackend = path.join(appDir, '..', 'main');
  let targetResourcesPath;
  
  console.log('After pack hook running...');
  console.log('Platform:', electronPlatformName);
  console.log('App output directory:', appOutDir);
  console.log('App directory:', appDir);
  
  // Determine target path based on platform
  if (electronPlatformName === 'darwin') {
    // macOS path
    targetResourcesPath = path.join(appOutDir, `${packager.appInfo.productName}.app`, 'Contents', 'Resources');
  } else if (electronPlatformName === 'win32') {
    // Windows path
    targetResourcesPath = path.join(appOutDir, 'resources');
  } else {
    // Linux path
    targetResourcesPath = path.join(appOutDir, 'resources');
  }
  
  console.log('Target resources path:', targetResourcesPath);
  
  // Ensure source binary exists
  if (!fs.existsSync(sourceBackend)) {
    console.error('❌ Source backend not found at:', sourceBackend);
    throw new Error(`Backend binary not found at: ${sourceBackend}`);
  } else {
    console.log('✅ Source backend found at:', sourceBackend);
    console.log('  File size:', fs.statSync(sourceBackend).size, 'bytes');
  }
  
  // Create bin directory in the resources folder
  const targetBinDir = path.join(targetResourcesPath, 'bin');
  if (!fs.existsSync(targetBinDir)) {
    console.log('Creating bin directory at:', targetBinDir);
    fs.mkdirSync(targetBinDir, { recursive: true });
  }
  
  // Copy backend binary to the bin directory
  const targetBackendPath = path.join(targetBinDir, 'main');
  console.log(`Copying backend from ${sourceBackend} to ${targetBackendPath}`);
  
  try {
    fs.copyFileSync(sourceBackend, targetBackendPath);
    fs.chmodSync(targetBackendPath, '755'); // Make executable
    console.log('✅ Backend binary successfully copied to:', targetBackendPath);
    
    // Verify the file exists and has the correct permissions
    const stats = fs.statSync(targetBackendPath);
    console.log('Backend binary size:', stats.size, 'bytes');
    console.log('Backend binary permissions:', stats.mode.toString(8));
    
    // List the files in the bin directory to confirm
    console.log('Contents of bin directory:');
    console.log(fs.readdirSync(targetBinDir));

    // For macOS, fix the libomp.dylib path using install_name_tool
    if (electronPlatformName === 'darwin') {
      const appPath = path.join(appOutDir, `${packager.appInfo.productName}.app`);
      const frameworksPath = path.join(appPath, 'Contents', 'Frameworks');
      const libompPath = path.join(frameworksPath, 'libomp.dylib');
      const onnxPath      = path.join(frameworksPath, 'libonnxruntime.dylib');

      
      if (fs.existsSync(libompPath)) {
        console.log('Fixing libomp.dylib path in backend executable...');
        try {
          // Get the current install name
          const { stdout: currentName } = await execAsync(`otool -L "${targetBackendPath}" | grep libomp`);
          console.log('Current libomp path:', currentName.trim());
          
          // Extract the current path from otool output
          const currentPath = currentName.trim().split(' ')[0];
          console.log('Extracted current path:', currentPath);
          
          // Use install_name_tool to change the path
          await execAsync(`install_name_tool -change "${currentPath}" "@executable_path/../../Frameworks/libomp.dylib" "${targetBackendPath}"`);
          console.log('✅ Successfully updated libomp.dylib path in backend executable');
          
          // Verify the change
          const { stdout: newName } = await execAsync(`otool -L "${targetBackendPath}" | grep libomp`);
          console.log('New libomp path:', newName.trim());
        } catch (error) {
          console.error('Error fixing libomp.dylib path:', error);
          throw error;
        }
      } else {
        console.warn('⚠️ libomp.dylib not found at:', libompPath);
      }
    }
  } catch (error) {
    console.error('Error copying backend binary:', error);
    throw error;
  }
  
  // Copy electron-is-dev module to ensure it's available
  try {
    console.log('Ensuring electron-is-dev module is available...');
    const sourceModule = path.join(appDir, 'node_modules', 'electron-is-dev');
    const targetModuleDir = path.join(targetResourcesPath, 'app', 'node_modules', 'electron-is-dev');
    
    if (fs.existsSync(sourceModule)) {
      // Create the target directory if it doesn't exist
      if (!fs.existsSync(path.dirname(targetModuleDir))) {
        fs.mkdirSync(path.dirname(targetModuleDir), { recursive: true });
      }
      
      // Copy the module directory
      copyRecursive(sourceModule, targetModuleDir);
      console.log('✅ electron-is-dev module copied successfully');
    } else {
      console.warn('⚠️ electron-is-dev module not found at', sourceModule);
    }
  } catch (error) {
    console.error('Error copying electron-is-dev module:', error);
    // Continue even if this fails, since we have a fallback in the code
  }

  // Ensure libomp.dylib is executable for ShipIt quarantine stripping
  if (electronPlatformName === 'darwin') {
    const appName = packager.appInfo.productFilename;
    const frameworksDir = path.join(appOutDir, `${appName}.app`, 'Contents', 'Frameworks');
    const libPath = path.join(frameworksDir, 'libomp.dylib');
    console.log('🔧 Setting permissions on libomp.dylib to 755:', libPath);
    try {
      fs.chmodSync(libPath, 0o755);
    } catch (err) {
      console.warn('⚠️ Could not chmod libomp.dylib:', err);
    }
  }

  console.log('After-pack hook completed successfully!');
}

/**
 * Helper function to recursively copy a directory
 */
function copyRecursive(src, dest) {
  if (!fs.existsSync(dest)) {
    fs.mkdirSync(dest, { recursive: true });
  }
  
  const entries = fs.readdirSync(src, { withFileTypes: true });
  
  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    
    if (entry.isDirectory()) {
      copyRecursive(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}
