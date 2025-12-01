"use client";

import { useState, useEffect } from "react";

interface User {
  id: number;
  email: string;
  username?: string;
  profile_picture?: string;
}

export default function UserAvatar() {
  const [user, setUser] = useState<User | null>(null);
  const [imageError, setImageError] = useState(false);

  const loadUserData = async () => {
    const userData = localStorage.getItem("user");
    if (userData) {
      const userInfo = JSON.parse(userData);
      
      // Try to fetch fresh user data from API to get updated profile picture
      try {
        const response = await fetch(`http://localhost:5000/api/profile/${userInfo.id}`);
        if (response.ok) {
          const freshUserData = await response.json();
          setUser(freshUserData);
          // Update localStorage with fresh data
          localStorage.setItem("user", JSON.stringify(freshUserData));
        } else {
          // Fallback to stored data if API fails
          setUser(userInfo);
        }
      } catch (error) {
        // Fallback to stored data if network fails
        setUser(userInfo);
      }
    }
  };

  useEffect(() => {
    loadUserData();
    
    // Listen for storage changes (when user data is updated elsewhere)
    const handleStorageChange = () => {
      loadUserData();
    };
    
    window.addEventListener('storage', handleStorageChange);
    
    // Custom event for when profile is updated
    const handleProfileUpdate = () => {
      loadUserData();
    };
    
    window.addEventListener('profileUpdated', handleProfileUpdate);
    
    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('profileUpdated', handleProfileUpdate);
    };
  }, []);

  if (!user) return null;

  const displayName = user.username || user.email.split("@")[0];
  const initials = (user.username || user.email).slice(0, 2).toUpperCase();

  return (
    <div className="flex items-center gap-2">
      {user.profile_picture && !imageError ? (
        <img
          src={`http://localhost:5000${user.profile_picture}`}
          alt="Profile"
          className="w-8 h-8 rounded-full object-cover border border-gray-300"
          onError={() => setImageError(true)}
          onLoad={() => setImageError(false)}
        />
      ) : (
        <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white text-xs font-semibold">
          {initials}
        </div>
      )}
      <span className="hidden sm:inline text-sm font-medium text-gray-700 dark:text-gray-300">
        {displayName}
      </span>
    </div>
  );
}
