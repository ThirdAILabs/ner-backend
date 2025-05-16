'use client';

import { useEffect, useState } from 'react';
import { X } from 'lucide-react';

interface ErrorPopupProps {
  autoCloseTime?: number; // Time in ms before auto-closing, 0 means no auto-close
}

interface ErrorState {
  show: boolean;
  message: string;
  status?: number;
}

export function ErrorPopup({ autoCloseTime = 5000 }: ErrorPopupProps) {
  const [error, setError] = useState<ErrorState>({
    show: false,
    message: '',
    status: undefined
  });

  useEffect(() => {
    // Function to handle the custom API error event
    const handleApiError = (event: Event) => {
      const customEvent = event as CustomEvent<{ message: string; status?: number }>;
      setError({
        show: true,
        message: customEvent.detail.message,
        status: customEvent.detail.status
      });
      
      // Auto-close the popup after specified time (if > 0)
      if (autoCloseTime > 0) {
        setTimeout(() => {
          setError((prev) => ({ ...prev, show: false }));
        }, autoCloseTime);
      }
    };

    // Add event listener for our custom event
    window.addEventListener('api-error', handleApiError);
    
    // Cleanup function
    return () => {
      window.removeEventListener('api-error', handleApiError);
    };
  }, [autoCloseTime]);

  // Don't render anything if no error to show
  if (!error.show) return null;

  return (
    <div className="fixed top-4 right-4 z-50 max-w-sm">
      <div className="bg-red-50 border-l-4 border-red-500 rounded-md shadow-md p-4 flex">
        <div className="flex-grow mr-2">
          <div className="flex items-center">
            <div className="text-red-600 text-sm font-medium">
              {error.status && <span className="font-bold mr-1">Error {error.status}:</span>}
              {error.message}
            </div>
          </div>
        </div>
        <button 
          onClick={() => setError((prev) => ({ ...prev, show: false }))}
          className="text-gray-400 hover:text-gray-500"
        >
          <X size={18} />
        </button>
      </div>
    </div>
  );
} 