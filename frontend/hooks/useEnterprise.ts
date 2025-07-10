import { useState, useEffect } from 'react';
import { nerService } from '@/lib/backend';

import { environment } from '@/lib/environment';

export function useEnterprise() {
  return {
    isEnterprise: environment.enterpriseMode,
    enterpriseLoading: false,
  };
  // const [isEnterprise, setIsEnterprise] = useState<boolean>(false);
  // const [loading, setLoading] = useState(true);

  // useEffect(() => {
  //   nerService
  //     .getEnterprise()
  //     .then((enterprise) => {
  //       if (
  //         typeof enterprise === 'object' &&
  //         enterprise !== null &&
  //         'IsEnterpriseMode' in enterprise
  //       ) {
  //         setIsEnterprise(Boolean((enterprise as any).IsEnterpriseMode));
  //       } else {
  //         setIsEnterprise(enterprise === true);
  //       }
  //     })
  //     .catch(() => setIsEnterprise(false))
  //     .finally(() => setLoading(false));
  // }, []);

  // return { isEnterprise, enterpriseLoading: loading };
}
