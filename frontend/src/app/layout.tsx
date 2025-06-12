"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { Geist, Geist_Mono } from "next/font/google";
import { Switch } from "@/components/ui/switch";
import { Toaster } from "sonner";
import "./globals.css";

/* fonts */
const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const onAuthPage = !["/login", "/register"].includes(pathname); // true when authenticated pages

  const [isDark, setIsDark] = useState(false);

  /* initial theme */
  useEffect(() => {
    const stored = localStorage.getItem("theme");
    if (!stored) localStorage.setItem("theme", "light");
    if (stored === "dark") {
      document.documentElement.classList.add("dark");
      setIsDark(true);
    }
  }, []);

  /* theme toggle */
  const toggleTheme = (val: boolean) => {
    setIsDark(val);
    if (val) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  };

  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-gray-50 text-gray-900 dark:bg-black dark:text-white`}
      >
        {/* header */}
        <header className="sticky top-0 z-50 bg-white dark:bg-neutral-900 shadow">
          <div className="container mx-auto flex items-center justify-between px-6 py-4">
            {/* left */}
            {onAuthPage ? (
              <div className="flex items-center gap-3">
                <Image src="/factuai-logo.png" alt="FactuAI" width={32} height={32} className="rounded" />
                <h1 className="text-xl font-semibold tracking-tight">FactuAI</h1>
              </div>
            ) : (
              <span className="text-lg font-medium text-muted-foreground">FactuAI</span>
            )}

            {/* right */}
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Switch checked={isDark} onCheckedChange={toggleTheme} />
                <span className="hidden md:inline text-sm text-muted-foreground">Dark mode</span>
              </div>
              {onAuthPage && (
                <button
                  onClick={() => {
                    localStorage.removeItem("user");
                    window.location.href = "/login";
                  }}
                  className="px-3 py-1 text-sm font-medium bg-red-500 hover:bg-red-600 text-white rounded"
                >
                  Logout
                </button>
              )}
            </div>
          </div>
        </header>

        {/* main */}
        <main
          className={
            onAuthPage
              ? "container mx-auto px-6 py-8 flex justify-center items-start min-h-screen"
              : "flex justify-center items-center min-h-screen px-4"
          }
        >
          {children}
        </main>

        <Toaster />
      </body>
    </html>
  );
}
