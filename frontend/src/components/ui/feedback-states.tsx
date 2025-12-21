'use client';

import { AlertTriangle, Loader2, X } from 'lucide-react';
import { ActiveModelDisplay } from '@/features/ai-providers';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/primitives';
import { Card, CardContent } from '@/components/ui/card';

// ========================================================================================
// PROCESSING STATUS COMPONENT (Simple inline status)
// ========================================================================================

interface ProcessingStatusProps {
  readonly isProcessing: boolean;
  readonly message?: string;
  readonly progress?: number;
  readonly className?: string;
}

export function ProcessingStatus({
  isProcessing,
  message = 'Processing...',
  progress,
  className = '',
}: ProcessingStatusProps) {
  if (!isProcessing) return null;

  return (
    <div className={`flex items-center gap-2 text-sm text-muted-foreground ${className}`}>
      <Loader2 className="h-4 w-4 animate-spin" />
      <span>{message}</span>
      {progress !== undefined && (
        <div className="flex-1 max-w-32">
          <Progress value={Math.min(100, Math.max(0, progress))} className="h-2" />
          <span className="text-xs text-muted-foreground">{Math.round(progress)}%</span>
        </div>
      )}
    </div>
  );
}

// ========================================================================================
// LOADING ANIMATION COMPONENT (Full loading state with model display)
// ========================================================================================

interface LoadingAnimationProps {
  readonly phase: string;
  readonly progress: number;
  readonly currentClaim?: number;
  readonly className?: string;
}

export function LoadingAnimation({
  phase,
  progress,
  currentClaim,
  className = '',
}: LoadingAnimationProps) {
  const safeProgress = Math.min(100, Math.max(0, progress));

  return (
    <Card className={`border border-border shadow-sm ${className}`}>
      <CardContent className="p-8">
        <div className="text-center space-y-6">
          <div className="flex justify-center">
            <Loader2 className="h-10 w-10 animate-spin text-primary" />
          </div>

          <div className="space-y-1">
            <p key={phase} className="text-lg font-medium text-foreground">
              {phase}
            </p>
            {currentClaim !== undefined && currentClaim > 0 && (
              <p className="text-sm text-muted-foreground">Processing claim {currentClaim}</p>
            )}
          </div>

          <div className="flex justify-center">
            <ActiveModelDisplay />
          </div>

          <div className="space-y-2">
            <Progress value={safeProgress} className="h-2" />
            <p className="text-xs text-muted-foreground">{Math.round(safeProgress)}% complete</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ========================================================================================
// ERROR STATE COMPONENT
// ========================================================================================

interface ErrorStateProps {
  readonly error: string;
  readonly onRetry?: () => void;
  readonly onClear?: () => void;
  readonly title?: string;
  readonly className?: string;
  readonly retryText?: string;
  readonly clearText?: string;
}

export function ErrorState({
  error,
  onRetry,
  onClear,
  title = 'Something went wrong',
  className = '',
  retryText = 'Try Again',
  clearText = 'Clear',
}: ErrorStateProps) {
  return (
    <Card
      className={`border border-destructive/20 bg-destructive/5 animate-in slide-in-from-top duration-500 ${className}`}
    >
      <CardContent className="p-8">
        <div className="text-center space-y-6">
          <div className="flex justify-center">
            <div className="w-16 h-16 bg-destructive/10 rounded-full flex items-center justify-center">
              <AlertTriangle className="h-8 w-8 text-destructive" />
            </div>
          </div>

          <div className="space-y-2">
            <h3 className="text-lg font-medium text-foreground">{title}</h3>
            <p className="text-sm text-muted-foreground max-w-md mx-auto leading-relaxed">
              {error}
            </p>
          </div>

          {(onRetry || onClear) && (
            <div className="flex gap-3 justify-center">
              {onRetry && (
                <Button onClick={onRetry} variant="destructive" className="gap-2">
                  <Loader2 className="h-4 w-4" />
                  {retryText}
                </Button>
              )}

              {onClear && (
                <Button onClick={onClear} variant="outline" className="gap-2">
                  <X className="h-4 w-4" />
                  {clearText}
                </Button>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// Default exports for backwards compatibility
export default LoadingAnimation;
