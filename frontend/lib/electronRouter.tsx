// This utility helps manage routing in Electron environment
import { useEffect, useState } from 'react';

type DeploymentIds = string[];

// Hook to fetch deployment IDs from Electron main process
export function useDeploymentIds(): { deploymentIds: DeploymentIds; isLoading: boolean; error: string | null } {
  const [deploymentIds, setDeploymentIds] = useState<DeploymentIds>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchDeploymentIds() {
      try {
        // Check if running in Electron
        if (typeof window !== 'undefined' && 'electron' in window) {
          // @ts-ignore - Electron API is injected via preload script
          const ids = await window.electron.getDeploymentIds();
          setDeploymentIds(ids);
        } else {
          // Fallback for development in browser
          setDeploymentIds(['deployment1', 'deployment2', 'deployment3']);
        }
      } catch (err) {
        console.error('Failed to fetch deployment IDs:', err);
        setError('Failed to fetch deployment IDs. Please check your connection.');
        // Fallback IDs for error case
        setDeploymentIds(['deployment1']);
      } finally {
        setIsLoading(false);
      }
    }

    fetchDeploymentIds();
  }, []);

  return { deploymentIds, isLoading, error };
}

// Add type definition to window object for Electron APIs
declare global {
  interface Window {
    electron?: {
      getDeploymentIds: () => Promise<string[]>;
      // Add other Electron API methods as needed
    };
  }
} 