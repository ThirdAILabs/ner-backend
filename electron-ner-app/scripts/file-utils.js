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

export const openFileChooser = async (supportedTypes, isDirectoryMode = false, isCombinedMode = false) => {
  const result = {
    directlySelected: [],
    allFilePaths: [],
    allFilesMeta: [],
    totalSize: 0,
  }

  // Separate dialogs for file vs directory selection
  let dialogProperties;
  let dialogResult;

  if (isCombinedMode && process.platform === 'darwin') {
    // macOS combined mode - allows selecting both files and folders
    dialogProperties = ['openFile', 'openDirectory', 'multiSelections', 'treatPackageAsDirectory'];
    dialogResult = await dialog.showOpenDialog({
      filters: [
        {
          name: 'Supported Files',
          extensions: supportedTypes
        },
      ],
      properties: dialogProperties,
      buttonLabel: 'Select Files or Folders',
      title: 'Select files or folders to scan'
    });
  } else if (isDirectoryMode) {
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

  if (dialogResult.canceled) return result;

  let allFilePaths = await gatherFilesRecursively(dialogResult.filePaths, supportedTypes);
  // Deduplicate allFiles and sort alphabetically
  // Only deduplicates by the file path
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

  result.directlySelected = dialogResult.filePaths;
  result.allFilePaths = allFilePaths;
  result.allFilesMeta = allFilesMeta;
  result.totalSize = allFilesMeta.reduce((sum, file) => sum + file.size, 0);
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

export const showFileInFolder = (filePath) => {
  try {
    shell.showItemInFolder(filePath);
  } catch (err) {
    console.error('Failed to show file in folder', err);
    throw err;
  }

}
