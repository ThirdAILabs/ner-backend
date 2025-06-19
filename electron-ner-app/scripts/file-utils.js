import { dialog, shell } from 'electron';
import path from 'node:path';
import fs from 'node:fs';

const getFilesFromDirectory = async (dirPath, supportedTypes) => {
  const files = [];
  const entries = await fs.promises.readdir(dirPath, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dirPath, entry.name);
    if (entry.isDirectory()) {
      files.push(...await getFilesFromDirectory(fullPath, supportedTypes));
    } else {
      // Only include files with supported extensions
      const ext = path.extname(entry.name).toLowerCase().slice(1); // Remove the dot
      if (supportedTypes.includes(ext)) {
        files.push(fullPath);
      }
    }
  }
  return files;
};

const gatherFilesRecursively = async (filePaths, supportedTypes) => {
  const files = [];
  for (const selectedPath of filePaths) {
    const stats = await fs.promises.stat(selectedPath);
    if (stats.isDirectory()) {
      const filesFromDir = await getFilesFromDirectory(selectedPath, supportedTypes);
      files.push(...filesFromDir);
    } else {
      files.push(selectedPath);
    }
  }
  return files;
}

const pathToFile = async (filePath) => {
  const stats = await fs.promises.stat(filePath);
  const buffer = await fs.promises.readFile(filePath);
  
  return {
    name: path.basename(filePath),
    size: stats.size,
    type: 'application/octet-stream',
    lastModified: stats.mtimeMs,
    // Convert buffer to base64 for transmission
    data: buffer.toString('base64')
  };
};

export const openFileChooser = async (supportedTypes) => {
  const result = {
    directlySelected: [],
    allFilePaths: [],
    allFilesMeta: [],
    totalSize: 0,
  }

  const dialogResult = await dialog.showOpenDialog({
    filters: [
      { name: 'Supported Files', extensions: supportedTypes },
    ],
    properties: [
      // Note: we cannot both have openFile and openDirectory on Windows.
      'openFile', 
      'openDirectory', 
      'multiSelections',
    ]
  });

  if (dialogResult.canceled) return result;

  let allFilePaths = await gatherFilesRecursively(dialogResult.filePaths, supportedTypes);
  // Deduplicate allFiles and sort alphabetically
  // Only deduplicates by the file path
  allFilePaths = [...new Set(allFilePaths)].sort();


  let allFilesMeta = await Promise.all(
  allFilePaths.map(async (filePath) => {
    const stats = await fs.promises.stat(filePath);
    return {
      name: path.basename(filePath),
      size: stats.size,
      type: 'application/octet-stream',
      lastModified: stats.mtimeMs,
      fullPath: filePath,
    };
  })
);

// Deduplicate by name and size
const seen = new Set();
allFilesMeta = allFilesMeta.filter(file => {
  const key = `${file.name}:${file.size}`;
  if (seen.has(key)) return false;
  seen.add(key);
  return true;
});
  

  result.directlySelected = dialogResult.filePaths;
  result.allFilePaths = allFilePaths;
  result.allFilesMeta = allFilesMeta;
  result.totalSize = allFilesMeta.reduce((sum, file) => sum + file.size, 0);
  return result;
}

export const openFile = async (filePath) => {
  const error = await shell.openPath(filePath);
  if (error) {
    throw new Error(error);
  }
}
