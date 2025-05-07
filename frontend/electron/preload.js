const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld(
  'electron',
  {
    getDeploymentIds: () => ipcRenderer.invoke('get-deployment-ids'),
    // Add more methods as needed for your application
  }
); 