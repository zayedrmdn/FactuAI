'use client';

interface LoadingAnimationProps {
  readonly phase: string;
  readonly progress: number;
  readonly currentClaim?: number;
  readonly className?: string;
}

export default function LoadingAnimation({
  phase,
  progress,
  currentClaim,
  className = '',
}: Readonly<LoadingAnimationProps>) {
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

        {/* Current Claim Info */}
        {currentClaim && currentClaim > 0 && (
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

        {/* Optional Status Dots */}
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
