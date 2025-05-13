// preload.js
const { contextBridge, ipcRenderer } = require('electron');
const { FIXED_PORT } = require('./scripts/check-port');

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