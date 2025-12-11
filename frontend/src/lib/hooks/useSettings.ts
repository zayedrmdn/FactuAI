'use client';

import { useState, useEffect } from 'react';

interface Prefs {
  textSize: 'sm' | 'md' | 'lg';
}

const defaultPrefs: Prefs = { textSize: 'md' };

export function useSettings() {
  const [prefs, setPrefs] = useState<Prefs>(defaultPrefs);
  const [isDark, setIsDark] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);

  // Load preferences and theme from localStorage after component mounts
  useEffect(() => {
    // Load preferences
    const rawPrefs = localStorage.getItem('factuai_prefs');
    if (rawPrefs) {
      try {
        setPrefs({ ...defaultPrefs, ...JSON.parse(rawPrefs) });
      } catch (e) {
        console.warn('Failed to parse preferences:', e);
      }
    }

    // Load theme
    const stored = localStorage.getItem('theme') === 'dark';
    setIsDark(stored);
    document.documentElement.classList.toggle('dark', stored);

    setIsLoaded(true);
  }, []);

  const savePrefs = (p: Partial<Prefs>) => {
    if (!isLoaded) return;

    const merged = { ...prefs, ...p };
    setPrefs(merged);
    localStorage.setItem('factuai_prefs', JSON.stringify(merged));
  };

  const toggleTheme = (v: boolean) => {
    if (!isLoaded) return;

    setIsDark(v);
    document.documentElement.classList.toggle('dark', v);
    localStorage.setItem('theme', v ? 'dark' : 'light');
  };

  return {
    prefs,
    savePrefs,
    isDark,
    toggleTheme,
    isLoaded, // Export this so components can wait for hydration
  };
}
