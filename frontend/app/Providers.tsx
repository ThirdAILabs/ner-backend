'use client';

import { ReactNode } from 'react';
import { HealthProvider } from '@/contexts/HealthProvider';

interface ProvidersProps {
  children: ReactNode;
}

export function Providers({ children }: ProvidersProps) {
  return <HealthProvider>{children}</HealthProvider>;
}
