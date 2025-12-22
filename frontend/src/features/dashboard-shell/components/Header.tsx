'use client';

import { useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Menu, X } from 'lucide-react';
import { useUser } from '@/lib/hooks/useUser';
import { getMediaUrl } from '@/lib/apiBase';

interface HeaderProps {
  readonly collapsed?: boolean;
  readonly onMobileMenuToggle?: () => void;
  readonly mobileMenuOpen?: boolean;
}

export function Header({ onMobileMenuToggle, mobileMenuOpen = false }: Readonly<HeaderProps>) {
  const { user } = useUser();
  const [imageError, setImageError] = useState(false);

  const displayName = user?.username || user?.email?.split('@')[0] || 'User';
  const initials = displayName.slice(0, 2).toUpperCase();

  return (
    <div className="flex h-16 shrink-0 items-center justify-between border-b border-border bg-card px-3 sm:px-4 gap-2 sm:gap-4">
      <Button
        variant="ghost"
        size="icon"
        className="md:hidden shrink-0"
        onClick={onMobileMenuToggle}
        aria-label="Toggle menu"
      >
        {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
      </Button>

      <div className="hidden md:flex items-center gap-2 flex-1">
        {/* Title removed to avoid redundancy with Sidebar */}
      </div>

      <div className="flex items-center gap-2 sm:gap-4 shrink-0 ml-auto">
        <span className="hidden sm:inline text-sm font-medium text-muted-foreground">
          Welcome back, {displayName}
        </span>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="relative h-9 w-9 rounded-full p-0">
              {user?.profile_picture && !imageError ? (
                <Image
                  src={getMediaUrl(user.profile_picture)}
                  alt="Profile"
                  width={36}
                  height={36}
                  className="h-9 w-9 rounded-full object-cover border border-border"
                  onError={() => setImageError(true)}
                  unoptimized
                />
              ) : (
                <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary text-xs font-semibold text-primary-foreground">
                  {initials}
                </div>
              )}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-48">
            <DropdownMenuItem asChild>
              <Link href="/dashboard/profile" className="cursor-pointer">
                Profile Settings
              </Link>
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              className="text-destructive focus:text-destructive cursor-pointer"
              onClick={() => {
                localStorage.removeItem('user');
                globalThis.location.href = '/';
              }}
            >
              Log out
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  );
}
