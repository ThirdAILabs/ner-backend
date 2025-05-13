// preload.js
const { contextBridge, ipcRenderer } = require('electron');

// The fixed proxy port will always be 3099 (which forwards to the dynamic backend port)
const PROXY_PORT = 3099;
// Expose API information to the renderer process
contextBridge.exposeInMainWorld('electronAPI', {
  // Backend API connection information
  backendAPI: {
    baseUrl: `http://localhost:${PROXY_PORT}`,
    apiVersion: 'v1',
  },
  // You can add more IPC functions here if needed for direct Electron communication
});

// Log when preload script runs
console.log(`Preload script is configuring API endpoint: http://localhost:${PROXY_PORT}/api/v1`);

window.addEventListener('DOMContentLoaded', () => {
  const replaceText = (selector, text) => {
    const element = document.getElementById(selector);
    if (element) element.innerText = text;
  };

  for (const dependency of ['chrome', 'node', 'electron']) {
    replaceText(`${dependency}-version`, process.versions[dependency]);
  }
}); 