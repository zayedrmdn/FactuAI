'use client';

import Script from 'next/script';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Geist, Geist_Mono } from 'next/font/google';
import { motion, AnimatePresence } from 'framer-motion';
import { Toaster } from 'sonner';
import { ChevronDownIcon, MoonIcon, SunIcon } from 'lucide-react';
import UserAvatar from '@/components/UserAvatar';
import './globals.css';

/* fonts */
const geistSans = Geist({ variable: '--font-geist-sans', subsets: ['latin'] });
const geistMono = Geist_Mono({ variable: '--font-geist-mono', subsets: ['latin'] });

/** Helper function to get the main container className */
function getMainClassName(isHomePage: boolean, onAuthPage: boolean, isDashboard: boolean): string {
  if (isHomePage || isDashboard) {
    return ''; // Homepage and dashboard handle their own layout
  }
  if (onAuthPage) {
    return 'container mx-auto px-6 py-8 flex justify-center items-start min-h-screen';
  }
  return 'flex justify-center items-center min-h-screen px-4';
}

interface UserData {
  username?: string;
  email?: string;
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  const pathname = usePathname();
  const onAuthPage = !['/login', '/register'].includes(pathname); // true when authenticated pages
  const isHomePage = pathname === '/';
  const isDashboard = pathname?.startsWith('/dashboard');

  const [isDark, setIsDark] = useState(false);
  const [showProfileDropdown, setShowProfileDropdown] = useState(false);
  const [user, setUser] = useState<UserData | null>(null);

  /* initial theme */
  useEffect(() => {
    const stored = localStorage.getItem('theme');
    if (!stored) {
      localStorage.setItem('theme', 'light');
      setIsDark(false);
    } else if (stored === 'dark') {
      document.documentElement.classList.add('dark');
      setIsDark(true);
    } else {
      document.documentElement.classList.remove('dark');
      setIsDark(false);
    }
  }, []);

  /* listen for theme changes from other components */
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'theme') {
        const newTheme = e.newValue;
        if (newTheme === 'dark') {
          document.documentElement.classList.add('dark');
          setIsDark(true);
        } else {
          document.documentElement.classList.remove('dark');
          setIsDark(false);
        }
      }
    };

    // Listen for storage changes from other tabs/components
    globalThis.addEventListener('storage', handleStorageChange);

    // Also listen for custom theme change events within the same tab
    const handleThemeChange = (e: CustomEvent) => {
      const newTheme = e.detail.theme;
      if (newTheme === 'dark') {
        document.documentElement.classList.add('dark');
        setIsDark(true);
      } else {
        document.documentElement.classList.remove('dark');
        setIsDark(false);
      }
    };

    globalThis.addEventListener('themeChange', handleThemeChange as EventListener);

    return () => {
      globalThis.removeEventListener('storage', handleStorageChange);
      globalThis.removeEventListener('themeChange', handleThemeChange as EventListener);
    };
  }, []);

  /* load user data */
  useEffect(() => {
    const userData = localStorage.getItem('user');
    if (userData) {
      setUser(JSON.parse(userData));
    }
  }, []);

  /* close dropdown when clicking outside */
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Element;
      if (!target.closest('[data-dropdown]')) {
        setShowProfileDropdown(false);
      }
    };

    if (showProfileDropdown) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showProfileDropdown]);

  /* theme toggle */
  const toggleTheme = (val: boolean) => {
    setIsDark(val);
    const newTheme = val ? 'dark' : 'light';

    if (val) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }

    localStorage.setItem('theme', newTheme);

    // Dispatch custom event to sync with other components
    globalThis.dispatchEvent(
      new CustomEvent('themeChange', {
        detail: { theme: newTheme },
      })
    );
  };

  const handleLogout = () => {
    localStorage.removeItem('user');
    setShowProfileDropdown(false);
    globalThis.location.href = '/login';
  };

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
        {/* header */}
        {!isHomePage && !isDashboard && (
          <header className="sticky top-0 z-50 bg-white/95 dark:bg-neutral-900/95 backdrop-blur-md shadow border-b border-gray-200/50 dark:border-gray-700/50">
            <div className="max-w-7xl mx-auto px-4">
              <div className="flex items-center justify-between h-16">
                {/* Left - Logo */}
                <Link href="/" className="flex items-center gap-3 group">
                  <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center group-hover:scale-110 transition-all duration-200 shadow-lg">
                    <span className="font-bold text-white">🔍</span>
                  </div>
                  <span className="text-xl font-bold text-gray-900 dark:text-white tracking-tight">
                    FactuAI
                  </span>
                </Link>

                {/* Right - Grouped Controls */}
                <div className="flex items-center gap-3">
                  {/* Theme Toggle - Icon Only */}
                  <button
                    onClick={() => toggleTheme(!isDark)}
                    className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors duration-200"
                  >
                    <motion.div
                      initial={false}
                      animate={{
                        scale: isDark ? 1 : 0,
                        opacity: isDark ? 1 : 0,
                        rotate: isDark ? 0 : 180,
                      }}
                      transition={{ duration: 0.3 }}
                      className="absolute"
                    >
                      <MoonIcon className="w-5 h-5 text-gray-600 dark:text-gray-400" />
                    </motion.div>
                    <motion.div
                      initial={false}
                      animate={{
                        scale: isDark ? 0 : 1,
                        opacity: isDark ? 0 : 1,
                        rotate: isDark ? 180 : 0,
                      }}
                      transition={{ duration: 0.3 }}
                    >
                      <SunIcon className="w-5 h-5 text-gray-600 dark:text-gray-400" />
                    </motion.div>
                  </button>

                  {/* Auth Section */}
                  {onAuthPage && user && (
                    <div className="relative" data-dropdown>
                      <button
                        onClick={() => setShowProfileDropdown(!showProfileDropdown)}
                        className="flex items-center gap-2 p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors duration-200"
                      >
                        <UserAvatar />
                        <motion.div
                          animate={{ rotate: showProfileDropdown ? 180 : 0 }}
                          transition={{ duration: 0.2 }}
                        >
                          <ChevronDownIcon className="w-4 h-4 text-gray-500" />
                        </motion.div>
                      </button>

                      {/* Dropdown Menu */}
                      <AnimatePresence>
                        {showProfileDropdown && (
                          <motion.div
                            initial={{ opacity: 0, y: -10, scale: 0.95 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            exit={{ opacity: 0, y: -10, scale: 0.95 }}
                            transition={{ duration: 0.2 }}
                            className="absolute top-full right-0 mt-2 w-48 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 py-1"
                          >
                            <div className="px-3 py-2 border-b border-gray-200 dark:border-gray-700">
                              <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                                {user.username || user.email?.split('@')[0]}
                              </span>
                            </div>
                            <Link
                              href="/profile"
                              className="flex items-center px-3 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                              onClick={() => setShowProfileDropdown(false)}
                            >
                              Profile
                            </Link>
                            <Link
                              href="/dashboard"
                              className="flex items-center px-3 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                              onClick={() => setShowProfileDropdown(false)}
                            >
                              Dashboard
                            </Link>
                            <hr className="my-1 border-gray-200 dark:border-gray-700" />
                            <button
                              onClick={handleLogout}
                              className="w-full flex items-center px-3 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                            >
                              Logout
                            </button>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </header>
        )}

        {/* main */}
        {isDashboard ? (
          // Dashboard uses its own layout structure, no wrapper needed
          children
        ) : (
          <main className={getMainClassName(isHomePage, onAuthPage, isDashboard)}>{children}</main>
        )}

        <Toaster />
      </body>
    </html>
  );
}
