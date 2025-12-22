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

  // Handler to load history items via URL param
  const loadHistoryItem = (item: HistoryItem) => {
    router.push(`/dashboard?load_id=${item.id}`);
  };

  return (
    <TooltipProvider delayDuration={0}>
      <aside
        className={cn(
          'flex h-full shrink-0 flex-col border-r bg-sidebar',
          collapsed ? 'w-16' : 'w-72', // Expanded width increased for history content
          'transition-[width] duration-300 ease-in-out'
        )}
      >
        <div className="flex h-16 shrink-0 items-center justify-between border-b border-sidebar-border px-3">
          {!collapsed && (
            <div className="flex items-center gap-2 font-bold text-sidebar-foreground min-w-0">
              <ShieldCheck className="h-6 w-6 shrink-0 text-primary" />
              <span className="text-xl whitespace-nowrap">FactuAI</span>
            </div>
          )}

          <Button
            variant="ghost"
            size="icon"
            className={cn('text-sidebar-foreground', collapsed && 'mx-auto')}
            onClick={onToggle}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </Button>
        </div>

        <nav className="flex-1 overflow-y-auto overflow-x-hidden flex flex-col">
          <div className="p-2 space-y-1">
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

          {/* History Section (Visible only when expanded) */}
          {!collapsed && (
            <div className="mt-4 flex-1 flex flex-col min-h-0 border-t border-border/40">
              <div className="px-4 py-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                <History className="h-3.5 w-3.5" />
                History
              </div>
              <div className="flex-1 min-h-0 relative">
                {/* Reuse HistoryPanel directly, allowing it to handle scrolling */}
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
        </nav>

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
