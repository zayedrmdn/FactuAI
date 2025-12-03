"use client";

import { useState, useEffect, useCallback } from "react";

interface User {
  id: number;
  email: string;
  username?: string;
  profile_picture?: string;
}

// Global cache to prevent duplicate fetches
let userCache: User | null = null;
let userPromise: Promise<User | null> | null = null;

export function useUser() {
  const [user, setUser] = useState<User | null>(userCache);
  const [loading, setLoading] = useState(!userCache);

  const fetchUserData = useCallback(async (): Promise<User | null> => {
    // If we already have cached data, return it
    if (userCache) {
      return userCache;
    }

    // If a fetch is already in progress, wait for it
    if (userPromise) {
      return userPromise;
    }

    // Start a new fetch
    userPromise = (async () => {
      try {
        const userData = localStorage.getItem("user");
        if (!userData) {
          return null;
        }

        const userInfo = JSON.parse(userData);
        const response = await fetch(`/api/profile/${userInfo.id}`);

        if (response.ok) {
          const freshUserData = await response.json();
          // Update localStorage with fresh data
          localStorage.setItem("user", JSON.stringify(freshUserData));
          userCache = freshUserData;
          return freshUserData;
        } else {
          // Fall back to cached localStorage data
          return userInfo;
        }
      } catch (error) {
        console.warn("Failed to fetch user data:", error);
        // Fall back to localStorage data
        const userData = localStorage.getItem("user");
        return userData ? JSON.parse(userData) : null;
      } finally {
        userPromise = null;
      }
    })();

    return userPromise;
  }, []);

  useEffect(() => {
    if (!user) {
      fetchUserData().then((userData) => {
        setUser(userData);
        setLoading(false);
      });
    }
  }, [user, fetchUserData]);

  const refetch = useCallback(async () => {
    userCache = null; // Clear cache to force refetch
    setLoading(true);
    const userData = await fetchUserData();
    setUser(userData);
    setLoading(false);
    return userData;
  }, [fetchUserData]);

  return { user, loading, refetch };
}