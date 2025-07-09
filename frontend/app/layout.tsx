import './globals.css';
import { ErrorPopup } from '@/components/ui/ErrorPopup';
import { Providers } from './Providers';
import { DiscordButton } from '@/components/ui/DiscordButton';
import { WindowControls } from '@/components/ui/WindowControls';

export const metadata = {
  title: 'PocketShield',
  description: 'Democratize AI for everyone.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" data-theme="winter">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,200..800;1,200..800&display=swap"
          rel="stylesheet"
        />
        <link
          href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css"
          rel="stylesheet"
        />
      </head>
      {/* Top padding pushes the content down to make room for the title bar region of the electron app. */}
      <body className="flex min-h-screen w-full flex-col bg-white pt-[30px]">
        {/*
          Draggable region of the app window. titlebar class is defined in globals.css
          Ideally, we should use the same element to push the content down and make it draggable,
          but I couldn't get the styling right.
        */}
        <div className="fixed top-0 left-0 w-full h-[30px] titlebar" />
        <WindowControls />
        <Providers>
          <div className="flex flex-col min-h-[calc(100vh-30px)]">
            {/* Main content */}
            {children}
          </div>

          {/* Discord button fixed to bottom-left */}
          <div className="fixed bottom-[2px] left-[2px] z-50">
            <DiscordButton />
          </div>

          {/* Error popup */}
          <ErrorPopup autoCloseTime={7000} />
        </Providers>
      </body>
    </html>
  );
}
