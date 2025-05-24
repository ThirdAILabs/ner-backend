import './globals.css';
import { ErrorPopup } from '@/components/ui/ErrorPopup';
import { Providers } from './Providers';

export const metadata = {
  title: 'PocketShield',
  description: 'Democratize AI for everyone.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,200..800;1,200..800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="flex min-h-screen w-full flex-col bg-white pt-[20px]">
        <div className="fixed top-0 left-0 w-full h-[35px] titlebar"/>
        <Providers>
          {children}
          <ErrorPopup autoCloseTime={7000} />
        </Providers>
      </body>
    </html>
  );
}
