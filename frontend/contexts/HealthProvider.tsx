'use client';

import React, { createContext, useContext, useState, useEffect, useRef } from 'react';
import { nerService } from '@/lib/backend';
import { updateNerBaseUrl } from '@/lib/axios.config';

interface HealthContextType {
  healthStatus: boolean;
}

const HealthContext = createContext<HealthContextType>({
  healthStatus: false,
});

export function HealthProvider({ children }: { children: React.ReactNode }) {
  const [healthStatus, setHealthStatus] = useState<boolean>(false);
  const fails = useRef<number>(0);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const updated = await updateNerBaseUrl();
        if (!updated) {
          return false;
        }
        const response = await nerService.checkHealth();
        return response.status === 200;
      } catch (error) {
        return false;
      }
    };

    let timeoutId: NodeJS.Timeout;

    const pollHealth = async () => {
      const isHealthy = await checkHealth();
      if (!isHealthy) {
        fails.current++;
        if (fails.current >= 5) {
          console.error(`Health check failed ${fails.current} times`);
        }
        timeoutId = setTimeout(pollHealth, 1000);
      } else {
        setHealthStatus(true);
      }
    };

    pollHealth();

    return () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, []);

  return <HealthContext.Provider value={{ healthStatus }}>{children}</HealthContext.Provider>;
}

export function useHealth() {
  const context = useContext(HealthContext);
  if (context === undefined) {
    throw new Error('useHealth must be used within a HealthProvider');
  }
  return context;
}
