'use client';

import React, { createContext, useContext, useState, useEffect, useRef } from 'react';
import { nerService } from '@/lib/backend';
import { updateNerBaseUrl } from '@/lib/axios.config';

import Image from 'next/image';
import { keyframes } from '@emotion/react';
import styled from '@emotion/styled';
import { Typography } from '@mui/material';

const pulse = keyframes`
  0% {
    transform: scale(1);
    opacity: 0.8;
  }
  50% {
    transform: scale(1.1);
    opacity: 1;
  }
  100% {
    transform: scale(1);
    opacity: 0.8;
  }
`;

const LoadingContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-between;
  height: 100vh;
  padding: 25vh 0;
`;

const PulsingLogo = styled.div`
  animation: ${pulse} 2s ease-in-out infinite;
  margin: auto 0;
`;

const Loading = () => {
  return (
    <LoadingContainer>
      <PulsingLogo>
        <Image src="/thirdai-logo.png" alt="Logo" width={100} height={100} priority />
      </PulsingLogo>
      <Typography className="text-gray-500" variant="h6" sx={{ fontWeight: 600 }}>
        Warming up...
      </Typography>
    </LoadingContainer>
  );
};

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

  return (
    <HealthContext.Provider value={{ healthStatus }}>
      {healthStatus ? children : <Loading />}
    </HealthContext.Provider>
  );
}

export function useHealth() {
  const context = useContext(HealthContext);
  if (context === undefined) {
    throw new Error('useHealth must be used within a HealthProvider');
  }
  return context;
}
