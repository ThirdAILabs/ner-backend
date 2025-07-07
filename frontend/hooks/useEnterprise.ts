import { useState, useEffect } from 'react';
import { nerService } from '@/lib/backend';

export function useEnterprise() {
  const [isEnterprise, setIsEnterprise] = useState<boolean>(false);

  useEffect(() => {
    nerService
      .getEnterprise()
      .then((isEnterpriseMode) => setIsEnterprise(Boolean(isEnterpriseMode)))
      .catch((err) => {
        console.log('Failed to load enterprise information:', err);
        setIsEnterprise(false);
      });
  }, []);

  return {
    isEnterprise,
  };
}
