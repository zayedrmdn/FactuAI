'use client';

import { useState } from 'react';
import { useUser } from '@/lib/hooks/useUser';

export default function UserAvatar() {
  const { user } = useUser();
  const [imageError, setImageError] = useState(false);

  if (!user) return null;

  const displayName = user.username || user.email.split('@')[0];
  const initials = (user.username || user.email).slice(0, 2).toUpperCase();

  return (
    <div className="flex items-center gap-2">
      {user.profile_picture && !imageError ? (
        /* eslint-disable-next-line @next/next/no-img-element */
        <img
          src={`http://127.0.0.1:5000${user.profile_picture}`}
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
