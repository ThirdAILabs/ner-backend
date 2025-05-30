// preload.js
const { contextBridge, ipcRenderer } = require('electron');

window.addEventListener('DOMContentLoaded', () => {
  const replaceText = (selector, text) => {
    const element = document.getElementById(selector);
    if (element) element.innerText = text;
  };

  for (const dependency of ['chrome', 'node', 'electron']) {
    replaceText(`${dependency}-version`, process.versions[dependency]);
  }
});

// Expose telemetry API to renderer
contextBridge.exposeInMainWorld('electron', {
  sendTelemetry: (data) => ipcRenderer.invoke('telemetry', data),
  getUserId: () => ipcRenderer.invoke('get-user-id'),
  openFileChooser: (supportedTypes) => ipcRenderer.invoke('open-file-chooser', supportedTypes),
  openFile: (filePath) => ipcRenderer.invoke('open-file', filePath),
}); 