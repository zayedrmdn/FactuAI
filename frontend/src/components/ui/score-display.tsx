'use client';

import * as React from 'react';
import { Progress } from '@/components/ui/primitives';
import { cn } from '@/lib/utils';
import type { TextSize } from '@/types/dashboard/ui';

// ========================================================================================
// SHARED UTILITIES
// ========================================================================================

type ScoreVariant = NonNullable<React.ComponentProps<typeof Progress>['variant']>;

function clampPercent(value: number): number {
  if (Number.isNaN(value) || value === undefined) return 0;
  return Math.max(0, Math.min(100, value));
}

function getAIDetectionLabelAndVariant(score: number): {
  readonly label: string;
  readonly variant: ScoreVariant;
} {
  // Inverted scale: higher score means more likely AI-generated
  if (score >= 80) return { label: 'Very Likely AI', variant: 'destructive' };
  if (score >= 60) return { label: 'Likely AI', variant: 'warning' };
  if (score >= 40) return { label: 'Possibly AI', variant: 'warning' };
  if (score >= 20) return { label: 'Probably Human', variant: 'info' };
  return { label: 'Very Likely Human', variant: 'success' };
}

function getConfidenceLabelAndVariant(score: number): {
  readonly label: string;
  readonly variant: ScoreVariant;
} {
  // Normal scale: higher score means more confident
  if (score >= 80) return { label: 'Very High', variant: 'success' };
  if (score >= 60) return { label: 'High', variant: 'info' };
  if (score >= 40) return { label: 'Medium', variant: 'warning' };
  if (score >= 20) return { label: 'Low', variant: 'warning' };
  return { label: 'Very Low', variant: 'destructive' };
}

// ========================================================================================
// AI DETECTION SCORE COMPONENT
// ========================================================================================

interface AIDetectionScoreProps {
  readonly score: number;
  readonly error?: string | undefined;
  readonly className?: string;
  readonly textSize?: TextSize;
}

export function AIDetectionScore({
  score,
  error,
  className,
  textSize = 'md',
}: AIDetectionScoreProps) {
  const labelClass = {
    sm: 'text-[10px]',
    md: 'text-xs',
    lg: 'text-sm',
  }[textSize];

  const valueClass = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base',
  }[textSize];

  if (error) {
    return (
      <div className={cn('max-w-xs mx-auto mb-6', className)}>
        <div className="text-center text-sm text-muted-foreground p-4 border rounded-lg bg-card border-border">
          <p className="font-medium mb-1 text-foreground">AI Detection</p>
          <p className="text-xs text-destructive">{error}</p>
        </div>
      </div>
    );
  }

  const normalizedScore = clampPercent(score);
  const { label, variant } = getAIDetectionLabelAndVariant(normalizedScore);

  return (
    <div className={cn('w-full flex flex-col items-center', className)}>
      <div className="w-full max-w-xs text-center space-y-2">
        <div className={cn(labelClass, 'font-medium text-muted-foreground')}>
          AI Detection Score
        </div>
        <div className={cn(valueClass, 'font-semibold text-foreground')}>{label}</div>
        <div className={cn(labelClass, 'text-muted-foreground')}>{normalizedScore.toFixed(1)}%</div>
        <Progress value={normalizedScore} variant={variant} className="h-2" />
      </div>
    </div>
  );
}

interface OverallScoreProps {
  readonly score: number;
  readonly title?: string;
  readonly className?: string;
  readonly textSize?: TextSize;
}

export function OverallScore({
  score,
  title = 'Confidence Score',
  className,
  textSize = 'md',
}: OverallScoreProps) {
  const normalizedScore = clampPercent(score);
  const { label, variant } = getConfidenceLabelAndVariant(normalizedScore);

  const labelClass = {
    sm: 'text-[10px]',
    md: 'text-xs',
    lg: 'text-sm',
  }[textSize];

  const valueClass = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base',
  }[textSize];

  return (
    <div className={cn('w-full flex flex-col items-center', className)}>
      <div className="w-full max-w-xs text-center space-y-2">
        <div className={cn(labelClass, 'font-medium text-muted-foreground')}>{title}</div>
        <div className={cn(valueClass, 'font-semibold text-foreground')}>{label}</div>
        <div className={cn(labelClass, 'text-muted-foreground')}>{normalizedScore.toFixed(0)}%</div>
        <Progress value={normalizedScore} variant={variant} className="h-2" />
      </div>
    </div>
  );
}
// Default exports for backwards compatibility
export default OverallScore;
