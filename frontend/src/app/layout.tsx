import { Geist, Geist_Mono } from 'next/font/google';
import { Toaster } from 'sonner';
import ClientLayout from './ClientLayout';
import './globals.css';

/* fonts */
const geistSans = Geist({ variable: '--font-geist-sans', subsets: ['latin'] });
const geistMono = Geist_Mono({ variable: '--font-geist-mono', subsets: ['latin'] });

export const metadata = {
  title: 'FactuAI - AI-Powered News Summarization & Fact-Checking',
  description:
    'Cut through misinformation with AI-powered news summarization and fact-checking backed by real evidence.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        <ClientLayout>{children}</ClientLayout>
        <Toaster />
      </body>
    </html>
  );
}
