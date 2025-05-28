import { useEffect, useState } from 'react';
import { nerService } from '@/lib/backend';

export default function useApiKeyStore() {
  const [apiKey, setApiKey] = useState<string>('');

  const saveApiKey = async (key: string) => {
    try {
      await nerService.setOpenAIApiKey(key);
    } catch (error) {
      alert('Failed to set OpenAI API key. Please try again.');
      return;
    }

    setApiKey(key);
  };

  useEffect(() => {
    nerService
      .getOpenAIApiKey()
      .then((key) => {
        setApiKey(key);
      })
      .catch((error) => {
        alert('Failed to get OpenAI API key. Please try again.');
      });
  }, []);

  return {
    apiKey,
    saveApiKey,
  };
}
