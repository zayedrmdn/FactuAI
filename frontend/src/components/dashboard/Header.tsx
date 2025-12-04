"use client";

import { useState } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useUser } from "@/app/dashboard/hooks/useUser";

interface HeaderProps {
  readonly collapsed?: boolean;
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export function Header(_props: Readonly<HeaderProps>) {
  const { user } = useUser();
  const [imageError, setImageError] = useState(false);

  const displayName = user?.username || user?.email?.split("@")[0] || "User";
  const initials = displayName.slice(0, 2).toUpperCase();

  return (
    <div className="flex h-16 shrink-0 items-center justify-end border-b bg-white px-4">
       {/* User Profile */}
       <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-slate-600">
             Welcome back, {displayName}
          </span>
          
          <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              className="relative h-9 w-9 rounded-full p-0"
            >
              {user?.profile_picture && !imageError ? (
                <img
                  src={`http://localhost:5000${user.profile_picture}`}
                  alt="Profile"
                  className="h-9 w-9 rounded-full object-cover border border-slate-200"
                  onError={() => setImageError(true)}
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
                localStorage.removeItem("user");
                globalThis.location.href = "/";
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
