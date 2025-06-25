'use client';

import useTelemetry from '@/hooks/useTelemetry';

const handleClick = async (
  e: React.MouseEvent<HTMLButtonElement>,
  recordEvent: ReturnType<typeof useTelemetry>
) => {
  e.preventDefault();
  recordEvent({
    UserAction: 'Clicked Discord Button',
    UIComponent: 'Discord Button',
    Page: 'Pages with Discord Button',
  });
  try {
    // @ts-ignore
    if (window.electron?.openLinkExternally) {
      // @ts-ignore
      await window.electron.openLinkExternally('https://discord.gg/Eyx2ZT8K72');
    } else {
      // Fallback for non-electron environment
      window.open('https://discord.gg/Eyx2ZT8K72', '_blank');
    }
  } catch (error) {
    console.error('Failed to open Discord link:', error);
    // Fallback
    window.open('https://discord.gg/Eyx2ZT8K72', '_blank');
  }
};

export function DiscordButton() {
  const recordEvent = useTelemetry();

  return (
    <div className="absolute bottom-4 left-4 z-50">
      <button
        className="p-2 hover:bg-gray-100 rounded-full transition-colors duration-200 tooltip tooltip-right"
        data-tip="PocketLLM community"
        onClick={(e) => handleClick(e, recordEvent)}
        aria-label="Join Discord Community"
      >
        <i className="bi bi-discord text-2xl" style={{ color: '#7289da' }}></i>
      </button>
    </div>
  );
}
