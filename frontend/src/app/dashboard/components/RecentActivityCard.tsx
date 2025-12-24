// Path: frontend/src/app/dashboard/components/RecentActivityCard.tsx
'use client';

import { FileText, Image, Video } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { cn } from '@/lib/utils';
import type { HistoryItem } from '@/types/dashboard/factcheck';

interface RecentActivityCardProps {
  item: HistoryItem;
}

function getInputTypeIcon(type: string) {
  switch (type) {
    case 'image':
      return Image;
    case 'video':
      return Video;
    default:
      return FileText;
  }
}

function getStatusBadge(results: HistoryItem['results']) {
  if (!results || results.length === 0) {
    return { label: 'Pending', color: 'bg-muted text-muted-foreground border-border' };
  }

  const verdicts = results.map((r) => r.label?.toLowerCase());
  const hasTrue = verdicts.some((v) => v === 'true' || v === 'mostly_true');
  const hasFalse = verdicts.some((v) => v === 'false' || v === 'mostly_false');

  if (hasFalse && !hasTrue) {
    return { label: 'False', color: 'bg-destructive/10 text-destructive border-destructive/20' };
  }
  if (hasTrue && !hasFalse) {
    return { label: 'Verified', color: 'bg-success/10 text-success border-success/20' };
  }
  return { label: 'Mixed', color: 'bg-warning/10 text-warning border-warning/20' };
}

function formatTimeAgo(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays === 1) return 'Yesterday';
  return `${diffDays}d ago`;
}

export function RecentActivityCard({ item }: RecentActivityCardProps) {
  const router = useRouter();
  const Icon = getInputTypeIcon(item.type);
  const status = getStatusBadge(item.results);

  const handleClick = () => {
    router.push(`/dashboard?load_id=${item.id}`);
  };

  // Truncate input text for preview
  const previewText =
    (item.input?.length || 0) > 80
      ? item.input?.substring(0, 80) + '...'
      : item.input || 'No content';

  return (
    <button
      onClick={handleClick}
      className="w-full text-left bg-card border border-border rounded-xl p-4 hover:border-primary/30 hover:shadow-md transition-all duration-200 cursor-pointer group"
    >
      {/* Header Row */}
      <div className="flex justify-between items-start mb-3">
        <div className="p-2 rounded-lg bg-muted text-muted-foreground group-hover:text-foreground transition-colors">
          <Icon className="h-5 w-5" />
        </div>
        <span className={cn('px-2 py-1 rounded-full text-xs font-bold border', status.color)}>
          {status.label}
        </span>
      </div>

      {/* Content */}
      <h4 className="text-foreground font-medium mb-1 truncate">
        {item.type === 'text'
          ? 'Text Analysis'
          : item.input?.split('\n')[0]?.substring(0, 30) || 'Analysis'}
      </h4>
      <p className="text-muted-foreground text-sm line-clamp-2">{previewText}</p>

      {/* Footer */}
      <div className="mt-4 pt-3 border-t border-border flex justify-between items-center">
        <span className="text-xs text-muted-foreground/70">{formatTimeAgo(item.timestamp)}</span>
        <span className="text-primary text-xs font-semibold group-hover:underline">
          View Report
        </span>
      </div>
    </button>
  );
}
