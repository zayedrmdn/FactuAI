'use client';

import { ChevronDown, ChevronRight, FileText, Image as ImageIcon, Video, X } from 'lucide-react';
import { HistoryItem as HistoryItemType } from '@/types/dashboard/factcheck';

interface HistoryItemProps {
  item: HistoryItemType;
  isExpanded?: boolean;
  onToggleExpanded?: () => void;
  onLoad: () => void;
  onDelete: () => void;
}

const VERDICT_CONFIG = {
  true: { label: 'True', color: 'text-success bg-success/10 border-success/20' },
  mostly_true: { label: 'Mostly True', color: 'text-success bg-success/10 border-success/20' },
  false: { label: 'False', color: 'text-destructive bg-destructive/10 border-destructive/20' },
  mostly_false: {
    label: 'Mostly False',
    color: 'text-destructive bg-destructive/10 border-destructive/20',
  },
  half_true: { label: 'Half True', color: 'text-warning bg-warning/10 border-warning/20' },
  barely_true: { label: 'Barely True', color: 'text-warning bg-warning/10 border-warning/20' },
  unknown: { label: 'Unknown', color: 'text-muted-foreground bg-muted/40 border-border' },
};

/** Get the icon box CSS classes based on item type */
function getIconBoxClasses(type: string): string {
  if (type === 'image') {
    return 'bg-primary/10 text-primary border-primary/20';
  }
  if (type === 'video') {
    return 'bg-secondary text-secondary-foreground border-border';
  }
  return 'bg-muted text-muted-foreground border-border';
}

export default function HistoryItem({
  item,
  isExpanded = false,
  onToggleExpanded,
  onLoad,
  onDelete,
}: Readonly<HistoryItemProps>) {
  const getInputTypeIcon = () => {
    switch (item.type) {
      case 'image':
        return <ImageIcon className="h-4 w-4" />;
      case 'video':
        return <Video className="h-4 w-4" />;
      default:
        return <FileText className="h-4 w-4" />;
    }
  };

  const getOverallAssessment = () => {
    if (!item.results?.length) return null;

    // Normalize verdict strings defensively; avoid undefined .includes
    const verdicts = item.results
      .map((r) => (typeof r.label === 'string' ? r.label.toLowerCase() : ''))
      .filter((v) => v.length > 0);

    if (verdicts.length === 0) return VERDICT_CONFIG['unknown'];

    if (verdicts.some((v) => v.includes('false'))) return VERDICT_CONFIG['false'];
    if (verdicts.some((v) => v.includes('half') || v.includes('barely')))
      return VERDICT_CONFIG['half_true'];
    if (verdicts.every((v) => v.includes('true'))) return VERDICT_CONFIG['true'];

    return VERDICT_CONFIG['unknown'];
  };

  const assessment = getOverallAssessment();

  // Format Date: "Today at 10:30 AM" or "Dec 2"
  const dateObj = new Date(item.timestamp);
  const dateStr = dateObj.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  const timeStr = dateObj.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });

  return (
    <div
      className={`group relative border rounded-lg transition-all duration-200 ${isExpanded ? 'bg-muted/30 border-primary/20 shadow-sm' : 'bg-card border-border hover:border-primary/20'}`}
    >
      {/* Main Clickable Area - using button for accessibility */}
      <button
        type="button"
        className="w-full p-3 text-left cursor-pointer"
        onClick={onToggleExpanded}
      >
        <div className="flex items-start justify-between gap-3">
          {/* Icon Box */}
          <div
            className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-md border ${getIconBoxClasses(item.type)}`}
          >
            {getInputTypeIcon()}
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-medium text-muted-foreground">
                {dateStr} • {timeStr}
              </span>
              {assessment && (
                <span
                  className={`text-[10px] uppercase tracking-wider font-bold px-1.5 py-0.5 rounded border ${assessment.color}`}
                >
                  {assessment.label}
                </span>
              )}
            </div>

            <p className="text-sm font-medium text-foreground line-clamp-2 leading-snug">
              {item.input || 'No text content'}
            </p>
          </div>

          {/* Chevron */}
          <div className="text-muted-foreground/50 shrink-0">
            {isExpanded ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </div>
        </div>
      </button>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="px-3 pb-3 pt-0 animate-in fade-in slide-in-from-top-1 duration-200">
          <div className="border-t border-border/50 pt-3 mt-1 space-y-3">
            {/* Quick Stats */}
            <div className="flex gap-4 text-xs text-muted-foreground">
              <span className="flex items-center gap-1">
                <span className="font-semibold text-foreground">{item.results?.length || 0}</span>{' '}
                Claims
              </span>
              {item.type === 'image' && item.metadata?.aiScore && (
                <span>AI Score: {item.metadata.aiScore.toFixed(0)}%</span>
              )}
            </div>

            {/* Actions */}
            <div className="flex gap-2">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onLoad();
                }}
                className="flex-1 bg-primary text-primary-foreground hover:bg-primary/90 text-xs py-2 px-3 rounded-md font-medium transition-colors"
              >
                View Full Report
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete();
                }}
                className="bg-destructive/10 text-destructive hover:bg-destructive/20 p-2 rounded-md transition-colors"
                title="Delete from history"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
