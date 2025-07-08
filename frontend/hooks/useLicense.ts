import { useState, useEffect } from 'react';
import { nerService } from '@/lib/backend';

export function useLicense() {
  const [license, setLicense] = useState<License | null>(null);

  useEffect(() => {
    nerService
      .getLicense()
      .then((lic) => setLicense(lic))
      .catch((err) => {
        console.log('Failed to load license:', err);
        setLicense(null);
      });
  }, []);

  return {
    license,
    isFreeVersion: license?.LicenseInfo.LicenseType === 'free' || false,
  };
}
