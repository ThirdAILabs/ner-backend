import { useCallback } from 'react';

export type TelemetryEvent = {
  UserAction: string;
  UIComponent: string;
  UI: string;
  data?: any;
};

export type TelemetryEventPackage = {
  username: string;
  timestamp: string;
  user_machine: string;
  event: TelemetryEvent;
};

export default function useTelemetry() {
  const recordEvent = useCallback(async (eventType: TelemetryEvent) => {
    const username = typeof window !== 'undefined'
      ? localStorage.getItem('user_id') || 'anonymous'
      : 'anonymous';
    const timestamp = new Date().toISOString();
    const userMachine = typeof navigator !== 'undefined'
      ? navigator.userAgent
      : 'server';

    const telemetryPackage: TelemetryEventPackage = {
      username,
      timestamp,
      user_machine: userMachine,
      event: eventType,
    };

    try {
      // Check if we're in Electron
      const isElectron = typeof window !== 'undefined' && 
        (window as any).electronAPI !== undefined;

      if (isElectron) {
        // Use Electron IPC for telemetry
        (window as any).electronAPI.sendTelemetry(telemetryPackage);
      } else {
        // Use API route for web/dev mode
        const response = await fetch('/api/telemetry', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(telemetryPackage),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
      }
    } catch (error) {
      console.error('Telemetry insert error:', error);
    }
  }, []);

  return recordEvent;
} 