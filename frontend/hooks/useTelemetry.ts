import { useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';

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
    // Pseudonymous user ID persisted across sessions
    let username: string;
    if (typeof window !== 'undefined') {
      const storageKey = 'telemetry_user_id';
      const storedId = localStorage.getItem(storageKey);
      if (storedId) {
        username = storedId;
      } else {
        username = uuidv4();
        localStorage.setItem(storageKey, username);
      }
    } else {
      username = 'server';
    }
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
      const res = await fetch('/api/telemetry', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(telemetryPackage),
      });
      if (!res.ok) {
        console.error('Telemetry insert error:', await res.text());
      }
    } catch (err) {
      console.error('Telemetry insert error:', err);
    }

  }, []);

  return recordEvent;
} 