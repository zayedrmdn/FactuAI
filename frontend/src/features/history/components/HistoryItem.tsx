'use client';

import {
  DocumentTextIcon,
  PhotoIcon,
  VideoCameraIcon,
  XMarkIcon,
  ChevronDownIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline';
import { HistoryItem as HistoryItemType } from '@/types/dashboard/factcheck';

interface HistoryItemProps {
  item: HistoryItemType;
  isExpanded?: boolean;
  onToggleExpanded?: () => void;
  onLoad: () => void;
  onDelete: () => void;
}

const VERDICT_CONFIG = {
  true: { label: 'True', color: 'text-emerald-600 bg-emerald-50 border-emerald-200' },
  mostly_true: { label: 'Mostly True', color: 'text-emerald-600 bg-emerald-50 border-emerald-200' },
  false: { label: 'False', color: 'text-rose-600 bg-rose-50 border-rose-200' },
  mostly_false: { label: 'Mostly False', color: 'text-rose-600 bg-rose-50 border-rose-200' },
  half_true: { label: 'Half True', color: 'text-amber-600 bg-amber-50 border-amber-200' },
  barely_true: { label: 'Barely True', color: 'text-amber-600 bg-amber-50 border-amber-200' },
  unknown: { label: 'Unknown', color: 'text-slate-600 bg-slate-50 border-slate-200' },
};

/** Get the icon box CSS classes based on item type */
function getIconBoxClasses(type: string): string {
  if (type === 'image') {
    return 'bg-blue-50 text-blue-600 border-blue-100';
  }
  if (type === 'video') {
    return 'bg-purple-50 text-purple-600 border-purple-100';
  }
  return 'bg-slate-50 text-slate-600 border-slate-100';
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
        return <PhotoIcon className="w-4 h-4" />;
      case 'video':
        return <VideoCameraIcon className="w-4 h-4" />;
      default:
        return <DocumentTextIcon className="w-4 h-4" />;
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
              <ChevronDownIcon className="w-4 h-4" />
            ) : (
              <ChevronRightIcon className="w-4 h-4" />
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
                <XMarkIcon className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
