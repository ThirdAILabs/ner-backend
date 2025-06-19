'use client';

import React from 'react';
import { Minus, Square, X } from 'lucide-react';

export function WindowControls() {
  const handleMinimize = () => {
    if (window.electron?.minimizeWindow) {
      window.electron.minimizeWindow();
    }
  };

  const handleMaximize = () => {
    if (window.electron?.maximizeWindow) {
      window.electron.maximizeWindow();
    }
  };

  const handleClose = () => {
    if (window.electron?.closeWindow) {
      window.electron.closeWindow();
    }
  };

  return (
    <div className="fixed top-0 right-0 h-[30px] flex items-center gap-1 pr-2 z-50">
      <button
        onClick={handleMinimize}
        className="w-7 h-7 rounded-md hover:bg-gray-200 transition-colors flex items-center justify-center"
        style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}
      >
        <Minus className="w-4 h-4 text-gray-600" />
      </button>
      <button
        onClick={handleMaximize}
        className="w-7 h-7 rounded-md hover:bg-gray-200 transition-colors flex items-center justify-center"
        style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}
      >
        <Square className="w-3 h-3 text-gray-600" />
      </button>
      <button
        onClick={handleClose}
        className="w-7 h-7 rounded-md hover:bg-red-500 hover:text-white transition-colors flex items-center justify-center"
        style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}
      >
        <X className="w-4 h-4 text-gray-600" />
      </button>
    </div>
  );
}