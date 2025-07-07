import { useState, useEffect } from 'react';
import { nerService } from '@/lib/backend';

export function useEnterprise() {
  const [isEnterprise, setIsEnterprise] = useState<boolean>(false);

  useEffect(() => {
    nerService
      .getEnterprise()
      .then((enterprise) => {
        if (
          typeof enterprise === 'object' &&
          enterprise !== null &&
          'IsEnterpriseMode' in enterprise
        ) {
          setIsEnterprise(Boolean((enterprise as any).IsEnterpriseMode));
        } else {
          setIsEnterprise(enterprise === true);
        }
      })
      .catch((err) => {
        console.log('Failed to load enterprise information:', err);
        setIsEnterprise(false);
      });
  }, []);

  return {
    isEnterprise,
  };
}
