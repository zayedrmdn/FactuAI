'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { ChevronDownIcon, MoonIcon, Search, SunIcon } from 'lucide-react';
import { UserAvatar } from '@/features/auth';

export default function LandingNav() {
  const [isDark, setIsDark] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [showProfileDropdown, setShowProfileDropdown] = useState(false);
  const [user, setUser] = useState<{
    username?: string;
    email?: string;
    profile_picture?: string;
  } | null>(null);

  // Handle theme
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

  // Listen for theme changes from other components
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

    // Listen for storage changes from other tabs
    globalThis.addEventListener('storage', handleStorageChange);

    // Also listen for custom theme change events within the same tab
    const handleThemeChange = (e: CustomEvent) => {
      const newTheme = e.detail.theme;
      if (newTheme === 'dark') {
        setIsDark(true);
      } else {
        setIsDark(false);
      }
    };

    globalThis.addEventListener('themeChange', handleThemeChange as EventListener);

    return () => {
      globalThis.removeEventListener('storage', handleStorageChange);
      globalThis.removeEventListener('themeChange', handleThemeChange as EventListener);
    };
  }, []);

  // Check authentication status
  useEffect(() => {
    const userData = localStorage.getItem('user');
    if (userData) {
      setIsAuthenticated(true);
      setUser(JSON.parse(userData));
    } else {
      setIsAuthenticated(false);
      setUser(null);
    }

    // Listen for auth changes
    const handleStorageChange = () => {
      const userData = localStorage.getItem('user');
      if (userData) {
        setIsAuthenticated(true);
        setUser(JSON.parse(userData));
      } else {
        setIsAuthenticated(false);
        setUser(null);
      }
    };

    globalThis.addEventListener('storage', handleStorageChange);
    return () => globalThis.removeEventListener('storage', handleStorageChange);
  }, []);

  // Handle scroll effect
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Close dropdown when clicking outside
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
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    // Clear the user hook cache to prevent stale data
    import('@/lib/hooks/useUser').then(({ clearUserCache }) => clearUserCache());
    setIsAuthenticated(false);
    setUser(null);
    setShowProfileDropdown(false);
    globalThis.location.reload(); // Refresh to show landing page
  };

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
    setIsMobileMenuOpen(false);
  };

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        isScrolled
          ? 'bg-background/95 backdrop-blur-md shadow-lg border-b border-border/50'
          : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Left - Enhanced Logo with Icon */}
          <Link href="/" className="flex items-center gap-3 group">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center group-hover:scale-110 transition-all duration-200 shadow-sm">
              <Search className="w-4 h-4 text-primary-foreground" aria-hidden="true" />
            </div>
            <span className="text-xl font-bold text-foreground tracking-tight">FactuAI</span>
          </Link>

          {/* Center - Navigation Links with Animations (Desktop) */}
          <div className="hidden md:flex items-center space-x-8">
            {[
              { href: '#features', label: 'Features' },
              { href: '#how-it-works', label: 'How It Works' },
              { href: '#contact', label: 'Contact' },
            ].map((link) => (
              <button
                key={link.href}
                onClick={() => scrollToSection(link.href.replace('#', ''))}
                className="relative text-muted-foreground hover:text-foreground font-medium text-sm transition-colors duration-200 group"
              >
                {link.label}
                <span className="absolute inset-x-0 -bottom-1 h-0.5 bg-primary transform scale-x-0 group-hover:scale-x-100 transition-transform duration-300 ease-out"></span>
              </button>
            ))}
          </div>

          {/* Right - Grouped Controls */}
          <div className="flex items-center gap-3">
            {/* Theme Toggle - Icon Only */}
            <button
              onClick={() => toggleTheme(!isDark)}
              className="p-2 rounded-lg hover:bg-accent transition-colors duration-200"
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
                <MoonIcon className="w-5 h-5 text-muted-foreground" />
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
                <SunIcon className="w-5 h-5 text-muted-foreground" />
              </motion.div>
            </button>

            {/* Auth Section */}
            {isAuthenticated && user ? (
              <div className="hidden md:block relative" data-dropdown>
                <button
                  onClick={() => setShowProfileDropdown(!showProfileDropdown)}
                  className="flex items-center gap-2 p-2 rounded-lg hover:bg-accent transition-colors duration-200"
                >
                  <UserAvatar />
                  <motion.div
                    animate={{ rotate: showProfileDropdown ? 180 : 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    <ChevronDownIcon className="w-4 h-4 text-muted-foreground" />
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
                      className="absolute top-full right-0 mt-2 w-48 bg-popover rounded-lg shadow-xl border border-border py-1"
                    >
                      <div className="px-3 py-2 border-b border-border">
                        <span className="text-sm font-medium text-foreground">
                          {user.username || user.email?.split('@')[0]}
                        </span>
                      </div>
                      <Link
                        href="/profile"
                        className="flex items-center px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
                        onClick={() => setShowProfileDropdown(false)}
                      >
                        Profile
                      </Link>
                      <Link
                        href="/dashboard"
                        className="flex items-center px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
                        onClick={() => setShowProfileDropdown(false)}
                      >
                        Dashboard
                      </Link>
                      <hr className="my-1 border-border" />
                      <button
                        onClick={handleLogout}
                        className="w-full flex items-center px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
                      >
                        Logout
                      </button>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            ) : (
              <div className="hidden md:flex items-center gap-3">
                <Link href="/login">
                  <Button variant="outline" size="sm">
                    Login
                  </Button>
                </Link>
                <Link href="/register">
                  <Button size="sm">Register</Button>
                </Link>
              </div>
            )}

            {/* Mobile Menu Button */}
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="md:hidden p-2 text-muted-foreground hover:bg-accent rounded-lg transition-colors duration-200"
            >
              <motion.svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                animate={{ rotate: isMobileMenuOpen ? 90 : 0 }}
                transition={{ duration: 0.2 }}
              >
                {isMobileMenuOpen ? (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                ) : (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                )}
              </motion.svg>
            </button>
          </div>
        </div>

        {/* Enhanced Mobile Menu */}
        <AnimatePresence>
          {isMobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
              className="md:hidden bg-background border-t border-border"
            >
              <div className="px-4 py-4 space-y-3">
                {/* Navigation Links */}
                <div className="space-y-2">
                  {[
                    { href: '#features', label: 'Features' },
                    { href: '#how-it-works', label: 'How It Works' },
                    { href: '#contact', label: 'Contact' },
                  ].map((link, index) => (
                    <motion.button
                      key={link.href}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      onClick={() => scrollToSection(link.href.replace('#', ''))}
                      className="block w-full text-left px-3 py-2 text-muted-foreground hover:text-foreground hover:bg-accent rounded-lg transition-all duration-200"
                    >
                      {link.label}
                    </motion.button>
                  ))}
                </div>

                {/* Auth Section */}
                <div className="border-t border-border pt-4">
                  {isAuthenticated && user ? (
                    <div className="space-y-3">
                      <div className="flex items-center gap-3 px-3 py-2">
                        <UserAvatar />
                        <span className="text-sm font-medium text-foreground">
                          {user.username || user.email?.split('@')[0]}
                        </span>
                      </div>
                      <div className="space-y-1">
                        <Link
                          href="/profile"
                          className="block"
                          onClick={() => setIsMobileMenuOpen(false)}
                        >
                          <div className="w-full text-left px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground rounded-lg transition-colors">
                            Profile
                          </div>
                        </Link>
                        <Link
                          href="/dashboard"
                          className="block"
                          onClick={() => setIsMobileMenuOpen(false)}
                        >
                          <div className="w-full text-left px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground rounded-lg transition-colors">
                            Dashboard
                          </div>
                        </Link>
                        <button
                          onClick={handleLogout}
                          className="w-full text-left px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground rounded-lg transition-colors"
                        >
                          Logout
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex gap-3">
                      <Link href="/login" className="flex-1">
                        <Button variant="outline" size="sm" className="w-full">
                          Login
                        </Button>
                      </Link>
                      <Link href="/register" className="flex-1">
                        <Button size="sm" className="w-full">
                          Register
                        </Button>
                      </Link>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </nav>
  );
}
