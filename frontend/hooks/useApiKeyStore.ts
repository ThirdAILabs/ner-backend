import { useEffect, useState } from 'react';
import { nerService } from '@/lib/backend';

export default function useApiKeyStore() {
  const [apiKey, setApiKey] = useState<string>('');

  const getApiKey = async () => {
    const { apiKey, error } = await nerService.getOpenAIApiKey();
    if (error) {
      alert(error);
      return '';
    }
    console.log('API key is', apiKey);
    return apiKey;
  };

  const saveApiKey = async (key: string) => {
    const error = await nerService.setOpenAIApiKey(key);
    if (error) {
      alert(error);
      return;
    }
    setApiKey(key);
  };

  useEffect(() => {
    getApiKey().then((key) => {
      console.log('Setting API key', key);
      setApiKey(key);
    });
  }, []);

  return {
    apiKey,
    saveApiKey,
  };
}
