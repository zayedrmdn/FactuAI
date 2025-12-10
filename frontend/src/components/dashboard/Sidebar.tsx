'use client';

import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { cn } from '@/lib/utils';
import {
  LayoutDashboard,
  User,
  LogOut,
  ChevronLeft,
  ChevronRight,
  ShieldCheck,
  Activity,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

interface SidebarProps {
  readonly collapsed: boolean;
  readonly onToggle: () => void;
}

const navItems = [
  {
    title: 'Dashboard',
    href: '/dashboard',
    icon: LayoutDashboard,
  },
  {
    title: 'API Limits',
    href: '/dashboard/limits',
    icon: Activity,
  },
  {
    title: 'Profile',
    href: '/dashboard/profile',
    icon: User,
  },
];

export function Sidebar({ collapsed, onToggle }: Readonly<SidebarProps>) {
  const pathname = usePathname();
  const router = useRouter();

  const handleLogout = () => {
    localStorage.removeItem('user');
    router.push('/');
    globalThis.location.reload();
  };

  return (
    <TooltipProvider delayDuration={0}>
      <aside
        className={cn(
          'flex h-full shrink-0 flex-col border-r bg-sidebar',
          collapsed ? 'w-16' : 'w-64',
          'transition-[width] duration-200 ease-in-out'
        )}
      >
        {/* Logo Area */}
        <div className="flex h-16 shrink-0 items-center border-b border-sidebar-border px-2">
          {collapsed ? (
            // Collapsed state: Show only toggle button centered
            <Button
              variant="ghost"
              size="icon"
              className="mx-auto text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
              onClick={onToggle}
              aria-label="Expand sidebar"
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          ) : (
            // Expanded state: Show logo and toggle button
            <>
              <Link
                href="/dashboard"
                className="flex flex-1 items-center gap-2 font-bold text-sidebar-foreground min-w-0 px-2"
              >
                <ShieldCheck className="h-6 w-6 shrink-0 text-primary" />
                <span className="text-xl whitespace-nowrap overflow-hidden transition-opacity duration-200">
                  FactuAI
                </span>
              </Link>

              <Button
                variant="ghost"
                size="icon"
                className="shrink-0 text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
                onClick={onToggle}
                aria-label="Collapse sidebar"
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
            </>
          )}
        </div>

        {/* Navigation - grows to fill space */}
        <nav className="flex-1 overflow-y-auto p-2">
          <div className="space-y-1">
            {navItems.map((item) => {
              const isActive = pathname === item.href;
              const linkContent = (
                <Link
                  href={item.href}
                  className={cn(
                    'flex items-center rounded-md text-sm font-medium transition-colors duration-150',
                    collapsed ? 'justify-center px-2 py-2.5' : 'gap-3 px-3 py-2.5',
                    isActive
                      ? 'bg-sidebar-accent text-sidebar-accent-foreground'
                      : 'text-sidebar-foreground hover:bg-sidebar-accent/50 hover:text-sidebar-accent-foreground'
                  )}
                >
                  <item.icon className="h-5 w-5 shrink-0" />
                  {!collapsed && <span className="whitespace-nowrap">{item.title}</span>}
                </Link>
              );

              if (collapsed) {
                return (
                  <Tooltip key={item.href}>
                    <TooltipTrigger asChild>{linkContent}</TooltipTrigger>
                    <TooltipContent side="right" sideOffset={8}>
                      {item.title}
                    </TooltipContent>
                  </Tooltip>
                );
              }
              return <div key={item.href}>{linkContent}</div>;
            })}
          </div>
        </nav>

        {/* Footer - pinned to bottom */}
        <div className="shrink-0 border-t border-sidebar-border p-2">
          {collapsed ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="w-full text-sidebar-foreground hover:bg-destructive/10 hover:text-destructive transition-colors duration-150"
                  onClick={handleLogout}
                  aria-label="Log out"
                >
                  <LogOut className="h-5 w-5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="right" sideOffset={8}>
                Log out
              </TooltipContent>
            </Tooltip>
          ) : (
            <Button
              variant="ghost"
              className="w-full justify-start gap-3 text-sidebar-foreground hover:bg-destructive/10 hover:text-destructive transition-colors duration-150"
              onClick={handleLogout}
            >
              <LogOut className="h-5 w-5 shrink-0" />
              <span className="whitespace-nowrap">Log out</span>
            </Button>
          )}
        </div>
      </aside>
    </TooltipProvider>
  );
}
