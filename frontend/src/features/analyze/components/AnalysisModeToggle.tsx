'use client';

import { Zap, FlaskConical } from 'lucide-react';
import { usePipelineModelsStore, type AnalysisMode } from '@/features/ai-providers';
import { cn } from '@/lib/utils';

interface AnalysisModeToggleProps {
  className?: string;
}

/**
 * Analysis Mode Toggle Component
 *
 * Allows user to switch between Quick and Deep analysis modes:
 * - Quick: Single search (15 results), no strategist, no pivot. ~5-7s.
 * - Deep: Full 4-phase pipeline with multi-angle queries and pivot. ~10-16s.
 */
export function AnalysisModeToggle({ className }: AnalysisModeToggleProps) {
  const { analysisMode, setAnalysisMode } = usePipelineModelsStore();

  const modes: { id: AnalysisMode; label: string; icon: typeof Zap; description: string }[] = [
    {
      id: 'quick',
      label: 'Quick',
      icon: Zap,
      description: 'Fast check (~5s). Single search, 15 sources.',
    },
    {
      id: 'deep',
      label: 'Deep',
      icon: FlaskConical,
      description: 'Thorough (~15s). Multi-angle queries, pivot research.',
    },
  ];

  return (
    <div className={cn('flex items-center gap-1 rounded-lg bg-muted p-1', className)}>
      {modes.map((mode) => {
        const Icon = mode.icon;
        const isActive = analysisMode === mode.id;
        return (
          <button
            key={mode.id}
            type="button"
            onClick={() => setAnalysisMode(mode.id)}
            title={mode.description}
            className={cn(
              'flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
              isActive
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground hover:bg-background/50'
            )}
          >
            <Icon className="h-4 w-4" />
            <span>{mode.label}</span>
          </button>
        );
      })}
    </div>
  );
}
