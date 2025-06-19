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

export const openFileChooser = async (supportedTypes, isDirectoryMode = false) => {
  const result = {
    directlySelected: [],
    allFiles: [],
    allFilePaths: [],
  }

  // Separate dialogs for file vs directory selection
  let dialogProperties;
  let dialogResult;

  if (isDirectoryMode) {
    // Directory selection mode
    dialogProperties = ['openDirectory'];
    dialogResult = await dialog.showOpenDialog({
      properties: dialogProperties,
      buttonLabel: 'Select Folder',
      title: 'Select a folder to scan'
    });
  } else {
    // File selection mode
    dialogProperties = ['openFile', 'multiSelections'];
    dialogResult = await dialog.showOpenDialog({
      filters: [
        {
          name: 'Supported Files',
          extensions: supportedTypes
        },
      ],
      properties: dialogProperties,
      buttonLabel: 'Select Files',
      title: 'Select files to scan'
    });
  }

  if (dialogResult.canceled) {
    return result;
  }

  let allFilePaths = await gatherFilesRecursively(dialogResult.filePaths, supportedTypes);
  
  // Deduplicate allFiles and sort alphabetically
  allFilePaths = [...new Set(allFilePaths)].sort();

  // If no supported files found in directory, show a warning
  if (isDirectoryMode && allFilePaths.length === 0) {
    dialog.showMessageBox({
      type: 'warning',
      title: 'No supported files found',
      message: `No files with supported extensions (${supportedTypes.join(', ')}) were found in the selected directory.`,
      buttons: ['OK']
    });
    return result;
  }

  const allFiles = await Promise.all(
    allFilePaths.map(pathToFile)
  );

  result.directlySelected = dialogResult.filePaths;
  result.allFiles = allFiles;
  result.allFilePaths = allFilePaths;

  return result;
} 

export const openFile = async (filePath) => {
  try {
    // Normalize path for Windows
    const normalizedPath = process.platform === 'win32' 
      ? filePath.replace(/\//g, '\\') 
      : filePath;
    
    console.log('Opening file:', normalizedPath);
    
    // Check if file exists before trying to open
    if (!fs.existsSync(normalizedPath)) {
      throw new Error(`File not found: ${normalizedPath}`);
    }
    
    const error = await shell.openPath(normalizedPath);
    if (error) {
      console.error('Error opening file:', error);
      throw new Error(error);
    }
  } catch (err) {
    console.error('Failed to open file:', err);
    throw err;
  }
}
