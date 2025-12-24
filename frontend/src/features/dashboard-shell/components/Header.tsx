'use client';

import { useState, useEffect } from 'react';
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
import { Menu, X, ChevronRight, Bell, BookOpen } from 'lucide-react';
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
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const displayName = user?.username || user?.email?.split('@')[0] || 'User';
  const initials = displayName.slice(0, 2).toUpperCase();

  return (
    <div className="flex h-14 shrink-0 items-center justify-between border-b border-border bg-card/50 backdrop-blur-sm px-4 sm:px-6 gap-4">
      {/* Mobile menu toggle */}
      <Button
        variant="ghost"
        size="icon"
        className="md:hidden shrink-0 h-9 w-9"
        onClick={onMobileMenuToggle}
        aria-label="Toggle menu"
      >
        {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
      </Button>

      {/* Breadcrumb Navigation */}
      <div className="hidden md:flex items-center text-sm">
        <span className="text-muted-foreground">Console</span>
        <ChevronRight className="h-4 w-4 mx-2 text-muted-foreground/50" />
        <span className="font-medium text-foreground">Verification Suite</span>
      </div>

      {/* Right Actions */}
      <div className="flex items-center gap-2 sm:gap-3 shrink-0 ml-auto">
        {/* Notification Bell */}
        <Button
          variant="ghost"
          size="icon"
          className="relative h-9 w-9 text-muted-foreground hover:text-foreground"
        >
          <Bell className="h-4 w-4" />
          <span className="absolute top-1.5 right-1.5 h-2 w-2 bg-primary rounded-full border-2 border-card" />
        </Button>

        {/* Docs Button */}
        <Link href="/docs" className="hidden sm:block">
          <Button variant="secondary" size="sm" className="h-9 px-4 gap-2 font-semibold">
            <BookOpen className="h-4 w-4" />
            Docs
          </Button>
        </Link>

        {/* User Menu */}
        {mounted && (
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
              <div className="px-2 py-1.5">
                <p className="text-sm font-medium">{displayName}</p>
                <p className="text-xs text-muted-foreground truncate">{user?.email}</p>
              </div>
              <DropdownMenuSeparator />
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
        )}
      </div>
    </div>
  );
}
