'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';
import { nerService } from '@/lib/backend';

interface HealthContextType {
    healthStatus: boolean;
}

const HealthContext = createContext<HealthContextType>({
    healthStatus: false
});

export function HealthProvider({ children }: { children: React.ReactNode }) {
    const [healthStatus, setHealthStatus] = useState<boolean>(false);

    useEffect(() => {
        const checkHealth = async () => {
            try {
                const response = await nerService.checkHealth();
                return response.status === 200;
            } catch (error) {
                console.error('Health check failed:', error);
                return false;
            }
        };

        let timeoutId: NodeJS.Timeout;

        const pollHealth = async () => {
            const isHealthy = await checkHealth();
            if (!isHealthy) {
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
            {children}
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