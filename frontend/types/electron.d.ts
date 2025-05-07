// TypeScript declarations for Electron API
interface ElectronAPI {
  api: {
    send: (channel: string, data: any) => void;
    receive: (channel: string, func: (...args: any[]) => void) => void;
  };
  isElectron: boolean;
}

interface Window {
  electron?: ElectronAPI;
} 