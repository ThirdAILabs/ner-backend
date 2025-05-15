import { useCallback } from 'react';
import { supabase } from '@/lib/supabaseClient';

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

    const { error } = await supabase.from('telemetry_events')
      .insert([telemetryPackage]);
    if (error) console.error('Telemetry insert error:', error);
  }, []);

  return recordEvent;
} 