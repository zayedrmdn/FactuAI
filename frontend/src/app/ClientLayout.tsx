'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDownIcon, MoonIcon, SunIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import UserAvatar from '@/components/UserAvatar';

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

export default function ClientLayout({ children }: Readonly<{ children: React.ReactNode }>) {
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
      try {
        setUser(JSON.parse(userData));
      } catch (error) {
        console.error('Error parsing user data:', error);
        localStorage.removeItem('user');
      }
    }
  }, []);

  const toggleTheme = (dark: boolean) => {
    setIsDark(dark);
    const newTheme = dark ? 'dark' : 'light';
    localStorage.setItem('theme', newTheme);

    if (dark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }

    // Dispatch custom event for other components in the same tab
    globalThis.dispatchEvent(new CustomEvent('themeChange', { detail: { theme: newTheme } }));
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
    setShowProfileDropdown(false);
    // Redirect to home page
    window.location.href = '/';
  };

  return (
    <>
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
                      rotate: isDark ? -180 : 0,
                    }}
                    transition={{ duration: 0.3 }}
                  >
                    <SunIcon className="w-5 h-5 text-gray-600 dark:text-gray-400" />
                  </motion.div>
                </button>

                {/* User Menu */}
                {user ? (
                  <div className="relative">
                    <button
                      onClick={() => setShowProfileDropdown(!showProfileDropdown)}
                      className="flex items-center gap-2 p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors duration-200"
                    >
                      <UserAvatar />
                      <ChevronDownIcon
                        className={`w-4 h-4 text-gray-600 dark:text-gray-400 transition-transform duration-200 ${
                          showProfileDropdown ? 'rotate-180' : ''
                        }`}
                      />
                    </button>

                    {/* Profile Dropdown */}
                    <AnimatePresence>
                      {showProfileDropdown && (
                        <motion.div
                          initial={{ opacity: 0, y: -10 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -10 }}
                          transition={{ duration: 0.2 }}
                          className="absolute right-0 mt-2 w-48 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 py-1 z-50"
                        >
                          <div className="px-4 py-2 border-b border-gray-200 dark:border-gray-700">
                            <p className="text-sm font-medium text-gray-900 dark:text-white">
                              {user.username || user.email}
                            </p>
                          </div>
                          <Link
                            href="/dashboard/profile"
                            className="block px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                            onClick={() => setShowProfileDropdown(false)}
                          >
                            Profile Settings
                          </Link>
                          <button
                            onClick={handleLogout}
                            className="block w-full text-left px-4 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-gray-100 dark:hover:bg-gray-700"
                          >
                            Sign Out
                          </button>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <Link href="/login">
                      <Button variant="ghost" size="sm">
                        Sign In
                      </Button>
                    </Link>
                    <Link href="/register">
                      <Button variant="default" size="sm">
                        Sign Up
                      </Button>
                    </Link>
                  </div>
                )}
              </div>
            </div>
          </div>
        </header>
      )}

      {/* main content */}
      <main className={getMainClassName(isHomePage, onAuthPage, isDashboard)}>
        {children}
      </main>
    </>
  );
}