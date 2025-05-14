// preload.js
import { contextBridge, ipcRenderer } from 'electron';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { FIXED_PORT } from './scripts/check-port.js';

// Get __dirname equivalent in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Expose API information to the renderer process
contextBridge.exposeInMainWorld('electronAPI', {
  // Backend API connection information
  backendAPI: {
    baseUrl: `http://localhost:${FIXED_PORT}`,  // Use our fixed port for backend URL
    apiVersion: 'v1',
  },
  // You can add more IPC functions here if needed for direct Electron communication
});

window.addEventListener('DOMContentLoaded', () => {
  const replaceText = (selector, text) => {
    const element = document.getElementById(selector);
    if (element) element.innerText = text;
  };

  for (const dependency of ['chrome', 'node', 'electron']) {
    replaceText(`${dependency}-version`, process.versions[dependency]);
  }
}); 