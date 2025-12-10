'use client';

import { useKeyLimits } from '@/lib/hooks/useKeyLimits';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { RefreshCw, TrendingUp, TrendingDown, DollarSign, Clock, Zap, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

/**
 * Formats credits to a readable string with commas
 */
function formatCredits(value: number | null): string {
  if (value === null) return 'Unlimited';
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 4,
  }).format(value);
}

/**
 * Formats percentage with color coding
 */
function formatPercentage(used: number, total: number | null): { text: string; color: string } {
  if (total === null) return { text: '0%', color: 'text-muted-foreground' };
  
  const percentage = (used / total) * 100;
  
  let color = 'text-green-600 dark:text-green-400';
  if (percentage > 80) color = 'text-red-600 dark:text-red-400';
  else if (percentage > 60) color = 'text-orange-600 dark:text-orange-400';
  else if (percentage > 40) color = 'text-yellow-600 dark:text-yellow-400';
  
  return {
    text: `${percentage.toFixed(1)}%`,
    color,
  };
}

/**
 * Progress bar component
 */
function UsageBar({ used, total, label }: Readonly<{ used: number; total: number | null; label: string }>) {
  const percentage = total === null ? 0 : Math.min((used / total) * 100, 100);
  
  let barColor = 'bg-green-500 dark:bg-green-400';
  if (percentage > 80) barColor = 'bg-red-500 dark:bg-red-400';
  else if (percentage > 60) barColor = 'bg-orange-500 dark:bg-orange-400';
  else if (percentage > 40) barColor = 'bg-yellow-500 dark:bg-yellow-400';
  
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs sm:text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium tabular-nums">
          {formatCredits(used)} {total !== null && `/ ${formatCredits(total)}`}
        </span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-secondary">
        <div
          className={cn('h-full transition-all duration-500', barColor)}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

/**
 * Metric card component
 */
function MetricCard({
  title,
  value,
  icon: Icon,
  trend,
}: Readonly<{
  title: string;
  value: string;
  icon: React.ElementType;
  trend?: 'up' | 'down' | 'neutral' | undefined;
}>) {
  return (
    <Card className="overflow-hidden">
      <CardContent className="p-4 sm:p-5">
        <div className="flex items-start justify-between gap-2">
          <div className="space-y-1.5 min-w-0 flex-1">
            <p className="text-xs sm:text-sm text-muted-foreground line-clamp-1">{title}</p>
            <p className="text-xl sm:text-2xl font-bold tabular-nums break-all">{value}</p>
          </div>
          <div className="shrink-0">
            <Icon className="h-5 w-5 sm:h-6 sm:w-6 text-muted-foreground" />
          </div>
        </div>
        {trend && (
          <div className="mt-2 flex items-center gap-1 text-xs">
            {trend === 'up' && <TrendingUp className="h-3 w-3 text-green-600 dark:text-green-400" />}
            {trend === 'down' && <TrendingDown className="h-3 w-3 text-red-600 dark:text-red-400" />}
            <span className={trend === 'up' ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}>
              {trend === 'up' ? 'Increasing' : 'Decreasing'}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * Loading skeleton
 */
function LoadingSkeleton() {
  const skeletonIds = ['remaining', 'limit', 'usage', 'reset'];
  
  return (
    <div className="space-y-4 sm:space-y-6">
      <div className="grid gap-3 sm:gap-4 grid-cols-2 lg:grid-cols-4">
        {skeletonIds.map((id) => (
          <Card key={`skeleton-${id}`}>
            <CardContent className="p-4 sm:p-5">
              <Skeleton className="h-12 sm:h-16 w-full" />
            </CardContent>
          </Card>
        ))}
      </div>
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-40" />
        </CardHeader>
        <CardContent className="space-y-4">
          <Skeleton className="h-16 w-full" />
          <Skeleton className="h-16 w-full" />
        </CardContent>
      </Card>
    </div>
  );
}

export default function LimitsPage() {
  const { data, error, loading, refetch } = useKeyLimits();

  return (
    <div className="container mx-auto max-w-7xl space-y-4 sm:space-y-6 p-4 sm:p-6 lg:p-8">
      {/* Header */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="space-y-1">
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">API Rate Limits</h1>
          <p className="text-sm text-muted-foreground">
            {data?.data.label ? `Key: ${data.data.label} • ` : ''}Monitor your OpenRouter API usage and quotas
          </p>
        </div>
        <Button
          onClick={() => refetch()}
          disabled={loading}
          size="sm"
          variant="outline"
          className="w-full sm:w-auto"
        >
          <RefreshCw className={cn('mr-2 h-4 w-4', loading && 'animate-spin')} />
          Refresh
        </Button>
      </div>

      {/* Error State */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Loading State */}
      {loading && !data && <LoadingSkeleton />}

      {/* Data Display */}
      {data && (
        <div className="space-y-4 sm:space-y-6">
          {/* Key Info Banner */}
          {data.data.is_free_tier && (
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription className="text-sm">
                You are on the <strong>free tier</strong>. Purchase credits to unlock higher rate limits.
              </AlertDescription>
            </Alert>
          )}

          {/* Quick Stats */}
          <div className="grid gap-3 sm:gap-4 grid-cols-2 lg:grid-cols-4">
            <MetricCard
              title="Remaining Credits"
              value={formatCredits(data.data.limit_remaining)}
              icon={DollarSign}
              trend={(() => {
                if (data.data.limit_remaining === null || data.data.limit === null) {
                  return undefined;
                }
                return data.data.limit_remaining / data.data.limit > 0.5 ? 'up' : 'down';
              })()}
            />
            <MetricCard
              title="Total Limit"
              value={formatCredits(data.data.limit)}
              icon={Zap}
            />
            <MetricCard
              title="Used (All Time)"
              value={formatCredits(data.data.usage)}
              icon={TrendingUp}
            />
            <MetricCard
              title="Reset Period"
              value={data.data.limit_reset || 'Never'}
              icon={Clock}
            />
          </div>

          {/* Credit Usage */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base sm:text-lg">Credit Usage</CardTitle>
              <CardDescription className="text-xs sm:text-sm">Standard API usage across different time periods</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <UsageBar
                used={data.data.usage_daily}
                total={data.data.limit}
                label="Today"
              />
              <UsageBar
                used={data.data.usage_weekly}
                total={data.data.limit}
                label="This Week"
              />
              <UsageBar
                used={data.data.usage_monthly}
                total={data.data.limit}
                label="This Month"
              />
              <UsageBar
                used={data.data.usage}
                total={data.data.limit}
                label="All Time"
              />
            </CardContent>
          </Card>

          {/* BYOK Usage */}
          {data.data.include_byok_in_limit && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base sm:text-lg">BYOK (Bring Your Own Key) Usage</CardTitle>
                <CardDescription className="text-xs sm:text-sm">External API usage included in your credit limit</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <UsageBar
                  used={data.data.byok_usage_daily}
                  total={data.data.limit}
                  label="Today"
                />
                <UsageBar
                  used={data.data.byok_usage_weekly}
                  total={data.data.limit}
                  label="This Week"
                />
                <UsageBar
                  used={data.data.byok_usage_monthly}
                  total={data.data.limit}
                  label="This Month"
                />
                <UsageBar
                  used={data.data.byok_usage}
                  total={data.data.limit}
                  label="All Time"
                />
              </CardContent>
            </Card>
          )}

          {/* Usage Summary Table */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base sm:text-lg">Usage Summary</CardTitle>
              <CardDescription className="text-xs sm:text-sm">Detailed breakdown of API usage</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto -mx-4 sm:mx-0">
                <div className="inline-block min-w-full align-middle">
                  <table className="min-w-full divide-y divide-border">
                    <thead>
                      <tr className="text-xs sm:text-sm">
                        <th className="px-4 py-2 text-left font-medium text-muted-foreground">Period</th>
                        <th className="px-4 py-2 text-right font-medium text-muted-foreground">Standard</th>
                        <th className="px-4 py-2 text-right font-medium text-muted-foreground">BYOK</th>
                        <th className="px-4 py-2 text-right font-medium text-muted-foreground">Total</th>
                        <th className="px-4 py-2 text-right font-medium text-muted-foreground">% of Limit</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border text-xs sm:text-sm">
                      {[
                        { label: 'Daily', standard: data.data.usage_daily, byok: data.data.byok_usage_daily },
                        { label: 'Weekly', standard: data.data.usage_weekly, byok: data.data.byok_usage_weekly },
                        { label: 'Monthly', standard: data.data.usage_monthly, byok: data.data.byok_usage_monthly },
                        { label: 'All Time', standard: data.data.usage, byok: data.data.byok_usage },
                      ].map((row) => {
                        const total = row.standard + row.byok;
                        const percentage = formatPercentage(total, data.data.limit);
                        return (
                          <tr key={row.label} className="hover:bg-muted/50 transition-colors">
                            <td className="px-4 py-2.5 font-medium">{row.label}</td>
                            <td className="px-4 py-2.5 text-right tabular-nums">{formatCredits(row.standard)}</td>
                            <td className="px-4 py-2.5 text-right tabular-nums">{formatCredits(row.byok)}</td>
                            <td className="px-4 py-2.5 text-right font-medium tabular-nums">{formatCredits(total)}</td>
                            <td className={cn('px-4 py-2.5 text-right font-medium tabular-nums', percentage.color)}>
                              {percentage.text}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Footer Info */}
          <div className="flex items-start gap-2 rounded-lg border border-border bg-muted/50 p-3 sm:p-4">
            <AlertCircle className="h-4 w-4 mt-0.5 shrink-0 text-muted-foreground" />
            <div className="space-y-1 text-xs sm:text-sm text-muted-foreground min-w-0">
              <p className="font-medium">Auto-refresh: Every 30 seconds</p>
              <p className="text-2xs sm:text-xs">Data is cached for 10 seconds on the server to prevent rate limiting.</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
