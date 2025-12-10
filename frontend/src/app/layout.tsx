import Script from 'next/script';
import { Geist, Geist_Mono } from 'next/font/google';
import { Toaster } from 'sonner';
import ClientLayout from './ClientLayout';
import './globals.css';

/* fonts */
const geistSans = Geist({ variable: '--font-geist-sans', subsets: ['latin'] });
const geistMono = Geist_Mono({ variable: '--font-geist-mono', subsets: ['latin'] });

export const metadata = {
  title: 'FactuAI - AI-Powered News Summarization & Fact-Checking',
  description: 'Cut through misinformation with AI-powered news summarization and fact-checking backed by real evidence.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <head>
        {/* React Grab: Development tool for AI-assisted development and faster context retrieval.
            Only loaded in development mode for security and bundle size reasons.
            See PROJECT_DOCUMENTATION.md for more details. */}
        {process.env.NODE_ENV === 'development' && (
          <Script
            src="//unpkg.com/react-grab/dist/index.global.js"
            crossOrigin="anonymous"
            strategy="beforeInteractive"
          />
        )}
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-gray-50 text-gray-900 dark:bg-black dark:text-white`}
      >
        <ClientLayout>{children}</ClientLayout>
        <Toaster />
      </body>
    </html>
  );
}