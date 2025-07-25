import { useCallback } from 'react';

export type TelemetryEvent = {
  UserAction: string;
  UIComponent: string;
  Page: string;
  Data?: any;
};

export type TelemetryEventPackage = {
  username: string;
  timestamp: string;
  user_machine: string;
  pocketshield_version: string | null;
  event: TelemetryEvent;
};

export default function useTelemetry() {
  const recordEvent = useCallback(async (eventType: TelemetryEvent) => {
    let username = 'anonymous';

    if (typeof window !== 'undefined') {
      // Check if we're in Electron and can get user ID
      const isElectron = (window as any).electron !== undefined;

      if (isElectron) {
        // Try to get user ID from localStorage first
        let storedUserId = localStorage.getItem('user_id');

        if (!storedUserId) {
          // If not in localStorage, get it from Electron and store it
          try {
            storedUserId = await (window as any).electron.getUserId();
            if (storedUserId && storedUserId !== 'anonymous') {
              localStorage.setItem('user_id', storedUserId);
            }
          } catch (error) {
            console.error('Error getting user ID from Electron:', error);
          }
        }

        username = storedUserId || 'anonymous';
      } else {
        // Web mode - use localStorage or fall back to anonymous
        username = localStorage.getItem('user_id') || 'anonymous';
      }
    }
    const timestamp = new Date().toISOString();
    const userMachine = typeof navigator !== 'undefined' ? navigator.userAgent : 'server';

    // Extract PocketShield version from user agent
    let pocketShieldVersion: string | null = null;
    if (userMachine && userMachine.includes('PocketShield/')) {
      const match = userMachine.match(/PocketShield\/([\d.]+)/);
      pocketShieldVersion = match ? match[1] : null;
    }

    const telemetryPackage: TelemetryEventPackage = {
      username,
      timestamp,
      user_machine: userMachine,
      pocketshield_version: pocketShieldVersion,
      event: eventType,
    };

    try {
      // Check if we're in Electron
      const isElectron = typeof window !== 'undefined' && (window as any).electron !== undefined;

      if (isElectron) {
        // Use Electron IPC for telemetry
        (window as any).electron.sendTelemetry(telemetryPackage);
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
