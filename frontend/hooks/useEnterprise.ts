import { useState, useEffect } from 'react';
import { nerService } from '@/lib/backend';

export function useEnterprise() {
  const [isEnterprise, setIsEnterprise] = useState<boolean>(false);
  const [loading, setLoading] = useState(true);

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
      .catch(() => setIsEnterprise(false))
      .finally(() => setLoading(false));
  }, []);

  return { isEnterprise, enterpriseLoading: loading };
}
