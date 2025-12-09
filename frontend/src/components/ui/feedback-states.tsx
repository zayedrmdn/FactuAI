'use client';

import { ArrowPathIcon, ExclamationTriangleIcon, XMarkIcon } from '@heroicons/react/24/outline';
import { ActiveModelDisplay } from '@/components/ai/ai-components';

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
      <ArrowPathIcon className="w-4 h-4 animate-spin" />
      <span>{message}</span>
      {progress !== undefined && (
        <div className="flex-1 max-w-32">
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
            />
          </div>
          <span className="text-xs text-gray-500">{Math.round(progress)}%</span>
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
  return (
    <div
      className={`relative p-8 bg-gradient-to-br from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 rounded-lg border border-purple-200 dark:border-purple-700 ${className}`}
    >
      <div className="absolute inset-0 bg-gradient-to-r from-purple-400/10 to-blue-400/10 rounded-lg animate-pulse" />

      <div className="relative z-10 text-center space-y-6">
        {/* Animated Spinner */}
        <div className="flex justify-center">
          <div className="relative w-16 h-16">
            <div className="absolute inset-0 border-4 border-purple-200 dark:border-purple-800 rounded-full" />
            <div className="absolute inset-0 border-4 border-purple-600 dark:border-purple-400 rounded-full border-t-transparent animate-spin" />
            <div
              className="absolute inset-2 border-2 border-blue-300 dark:border-blue-500 rounded-full border-b-transparent animate-spin"
              style={{ animationDirection: 'reverse', animationDuration: '1.5s' }}
            />
          </div>
        </div>

        {/* Phase Text with Animation */}
        <div className="h-8 flex items-center justify-center">
          <p
            key={phase}
            className="text-lg font-medium text-purple-700 dark:text-purple-300 animate-in slide-in-from-right-3 fade-in duration-500"
          >
            {phase}
          </p>
        </div>

        {/* Active Model Display */}
        <div className="flex justify-center">
          <ActiveModelDisplay />
        </div>

        {/* Current Claim Info */}
        {currentClaim !== undefined && currentClaim > 0 && (
          <p className="text-sm text-muted-foreground animate-in fade-in duration-300">
            Processing claim {currentClaim}
          </p>
        )}

        {/* Progress Bar */}
        <div className="space-y-2">
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-purple-600 to-blue-600 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
            />
          </div>
          <p className="text-xs text-muted-foreground">{Math.round(progress)}% complete</p>
        </div>

        {/* Status Dots */}
        <div className="flex justify-center space-x-1">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className={`w-2 h-2 rounded-full transition-all duration-300 ${
                progress > i * 33.33
                  ? 'bg-purple-600 dark:bg-purple-400'
                  : 'bg-gray-300 dark:bg-gray-600'
              }`}
              style={{
                animationDelay: `${i * 200}ms`,
                animation: progress > i * 33.33 ? 'pulse 2s infinite' : 'none',
              }}
            />
          ))}
        </div>
      </div>
    </div>
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
    <div
      className={`relative p-8 bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-lg border border-red-200 dark:border-red-700 animate-in slide-in-from-top duration-500 ${className}`}
    >
      <div className="text-center space-y-6">
        {/* Error Icon */}
        <div className="flex justify-center">
          <div className="w-16 h-16 bg-red-100 dark:bg-red-900/30 rounded-full flex items-center justify-center">
            <ExclamationTriangleIcon className="w-8 h-8 text-red-600 dark:text-red-400" />
          </div>
        </div>

        {/* Error Content */}
        <div className="space-y-2">
          <h3 className="text-lg font-medium text-red-700 dark:text-red-300">{title}</h3>
          <p className="text-sm text-red-600 dark:text-red-400 max-w-md mx-auto leading-relaxed">
            {error}
          </p>
        </div>

        {/* Action Buttons */}
        {(onRetry || onClear) && (
          <div className="flex gap-3 justify-center">
            {onRetry && (
              <button
                onClick={onRetry}
                className="flex items-center gap-2 bg-red-600 hover:bg-red-700 disabled:bg-red-400 text-white px-4 py-2 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2"
              >
                <ArrowPathIcon className="w-4 h-4" />
                {retryText}
              </button>
            )}

            {onClear && (
              <button
                onClick={onClear}
                className="flex items-center gap-2 text-red-600 hover:text-red-800 dark:text-red-400 dark:hover:text-red-200 px-4 py-2 rounded-lg transition-colors border border-red-300 hover:border-red-400 dark:border-red-600 dark:hover:border-red-500 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2"
              >
                <XMarkIcon className="w-4 h-4" />
                {clearText}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// Default exports for backwards compatibility
export default LoadingAnimation;
