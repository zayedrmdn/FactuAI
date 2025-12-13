'use client';

import { useState, useEffect, useCallback } from 'react';

interface User {
  id: number;
  email: string;
  username?: string;
  profile_picture?: string;
}

// Global cache to prevent duplicate state reads
let userCache: User | null = null;

/**
 * Clear the user cache - call this on logout to ensure fresh state
 */
export function clearUserCache(): void {
  userCache = null;
}

export function useUser() {
  const [user, setUser] = useState<User | null>(userCache);
  const [loading, setLoading] = useState(!userCache);

  const fetchUserData = useCallback((): User | null => {
    // If we already have cached data, return it
    if (userCache) {
      return userCache;
    }

    // Read from localStorage (the source of truth for auth state)
    // Backend V4 does not have a /api/profile endpoint, so we rely on localStorage
    try {
      const userData = localStorage.getItem('user');
      if (!userData) {
        return null;
      }

      const userInfo = JSON.parse(userData) as User;
      userCache = userInfo;
      return userInfo;
    } catch (error) {
      console.warn('Failed to parse user data from localStorage:', error);
      localStorage.removeItem('user');
      return null;
    }
  }, []);

  useEffect(() => {
    if (!user) {
      const userData = fetchUserData();
      setUser(userData);
      setLoading(false);
    }
  }, [user, fetchUserData]);

  const refetch = useCallback(() => {
    userCache = null; // Clear cache to force refresh from localStorage
    setLoading(true);
    const userData = fetchUserData();
    setUser(userData);
    setLoading(false);
    return userData;
  }, [fetchUserData]);

  return { user, loading, refetch };
}
