import { useState, useEffect } from 'react';
import { useLicense } from './useLicense';
import { useEnterprise } from './useEnterprise';

export function useFinetuning() {
  const [isFinetuningEnabled, setIsFinetuningEnabled] = useState<boolean>(false);
  const { isFreeVersion } = useLicense();
  const { isEnterprise } = useEnterprise();

  useEffect(() => {
    setIsFinetuningEnabled(!isFreeVersion && isEnterprise);
  }, [isFreeVersion, isEnterprise]);

  return {
    isFinetuningEnabled,
  };
}
