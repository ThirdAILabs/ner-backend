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