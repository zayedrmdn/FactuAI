// Path: frontend/src/features/dashboard-shell/components/Sidebar.tsx
'use client';

import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { cn } from '@/lib/utils';
import {
  ChevronLeft,
  ChevronRight,
  LayoutDashboard,
  LogOut,
  ShieldCheck,
  History,
  Settings,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { useHistory } from '@/features/history';
import { HistoryPanel } from '@/features/history';
import type { HistoryItem } from '@/types/dashboard/factcheck';

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
];

export function Sidebar({ collapsed, onToggle }: Readonly<SidebarProps>) {
  const pathname = usePathname();
  const router = useRouter();
  const { history, deleteHistoryItem, clearAllHistory } = useHistory();

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    import('@/lib/hooks/useUser').then(({ clearUserCache }) => clearUserCache());
    router.push('/');
    globalThis.location.reload();
  };

  const loadHistoryItem = (item: HistoryItem) => {
    router.push(`/dashboard?load_id=${item.id}`);
  };

  return (
    <TooltipProvider delayDuration={0}>
      <aside
        className={cn(
          'flex h-full shrink-0 flex-col border-r bg-sidebar overflow-hidden',
          collapsed ? 'w-16' : 'w-72',
          'transition-[width] duration-300 ease-in-out will-change-[width]'
        )}
      >
        {/* Header with Logo + Status */}
        <div className="flex h-16 shrink-0 items-center justify-between border-b border-sidebar-border px-3 overflow-hidden">
          <div
            className={cn(
              'flex items-center gap-3 min-w-0 transition-all duration-250 origin-left',
              collapsed
                ? 'opacity-0 -translate-x-2 scale-95 pointer-events-none'
                : 'opacity-100 translate-x-0 scale-100'
            )}
            aria-hidden={collapsed}
          >
            <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-primary/10 border border-primary/20 shadow-sm">
              <ShieldCheck className="h-5 w-5 text-primary" />
            </div>
            <div className="flex flex-col min-w-0">
              <span className="text-base font-bold text-sidebar-foreground tracking-tight">
                FactuAI
              </span>
              <div className="flex items-center gap-1.5">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-success opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-success" />
                </span>
                <span className="text-xs text-muted-foreground font-medium whitespace-nowrap">
                  Engine Online
                </span>
              </div>
            </div>
          </div>

          <Button
            variant="ghost"
            size="icon"
            className={cn(
              'text-sidebar-foreground hover:bg-sidebar-accent',
              collapsed && 'mx-auto'
            )}
            onClick={onToggle}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </Button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto overflow-x-hidden flex flex-col">
          <div className="p-2 space-y-1">
            {navItems.map((item) => {
              const isActive = pathname === item.href;
              const linkContent = (
                <Link
                  href={item.href}
                  className={cn(
                    'flex items-center rounded-lg text-sm font-medium transition-all duration-200',
                    collapsed ? 'justify-center px-2 py-2.5' : 'gap-3 px-3 py-2.5',
                    isActive
                      ? 'bg-primary/10 text-primary border border-primary/20 shadow-sm'
                      : 'text-sidebar-foreground hover:bg-sidebar-accent/50 hover:text-sidebar-accent-foreground'
                  )}
                >
                  <item.icon className={cn('h-5 w-5 shrink-0', isActive && 'text-primary')} />
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

          {/* History Section (Visible only when expanded) */}
          {!collapsed && (
            <div className="mt-4 flex-1 flex flex-col min-h-0 border-t border-border/40">
              <div className="px-4 py-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                <History className="h-3.5 w-3.5" />
                History
              </div>
              <div className="flex-1 min-h-0 relative">
                <div className="absolute inset-0">
                  <HistoryPanel
                    history={history}
                    load={loadHistoryItem}
                    del={deleteHistoryItem}
                    clearAll={clearAllHistory}
                  />
                </div>
              </div>
            </div>
          )}

          {/* System Section */}
          {!collapsed && (
            <div className="border-t border-border/40 p-2 space-y-1">
              <p className="px-3 py-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                System
              </p>
              <Link
                href="#"
                className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-sidebar-foreground hover:bg-sidebar-accent/50 hover:text-sidebar-accent-foreground transition-colors"
              >
                <Settings className="h-5 w-5" />
                <span>Settings</span>
              </Link>
            </div>
          )}
        </nav>

        {/* Footer - Logout */}
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
