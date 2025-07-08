// Platform utilities for cross-platform compatibility
const path = require('path');

function getPlatform() {
  return process.platform; // 'darwin', 'win32', 'linux'
}

function isWindows() {
  return process.platform === 'win32';
}

function isMacOS() {
  return process.platform === 'darwin';
}

function isLinux() {
  return process.platform === 'linux';
}

function getExecutableName(baseName) {
  return isWindows() ? `${baseName}.exe` : baseName;
}

function getOnnxRuntimeLibraryName() {
  if (isWindows()) return 'onnxruntime.dll';
  if (isMacOS()) return 'libonnxruntime.dylib';
  return 'libonnxruntime.so'; // Linux
}

function getWindowsDependencies() {
  if (!isWindows()) return [];
  
  return [
    'libgcc_s_seh-1.dll',
    'libstdc++-6.dll',
    'libwinpthread-1.dll',
    'onnxruntime.dll'
  ];
}

function getOnnxRuntimePath(isProduction, binPath) {
  if (isWindows()) {
    return path.join(binPath, 'onnxruntime.dll');
  } else if (isMacOS()) {
    if (isProduction) {
      const frameworksDir = path.join(path.dirname(process.execPath), '..', 'Frameworks');
      return path.join(frameworksDir, 'libonnxruntime.dylib');
    } else {
      return path.join(__dirname, '..', 'resources', 'libonnxruntime.dylib');
    }
  } else {
    // Linux
    return path.join(binPath, 'libonnxruntime.so');
  }
}

module.exports = {
  getPlatform,
  isWindows,
  isMacOS,
  isLinux,
  getExecutableName,
  getOnnxRuntimeLibraryName,
  getWindowsDependencies,
  getOnnxRuntimePath
};