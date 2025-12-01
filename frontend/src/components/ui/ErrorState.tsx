"use client";

import { ExclamationTriangleIcon, ArrowPathIcon, XMarkIcon } from "@heroicons/react/24/outline";

interface ErrorStateProps {
  error: string;
  onRetry?: () => void;
  onClear?: () => void;
  title?: string;
  className?: string;
  retryText?: string;
  clearText?: string;
}

export default function ErrorState({ 
  error, 
  onRetry, 
  onClear,
  title = "Something went wrong",
  className = "",
  retryText = "Try Again",
  clearText = "Clear"
}: ErrorStateProps) {
  return (
    <div className={`relative p-8 bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-lg border border-red-200 dark:border-red-700 animate-in slide-in-from-top duration-500 ${className}`}>
      <div className="text-center space-y-6">
        {/* Error Icon */}
        <div className="flex justify-center">
          <div className="w-16 h-16 bg-red-100 dark:bg-red-900/30 rounded-full flex items-center justify-center">
            <ExclamationTriangleIcon className="w-8 h-8 text-red-600 dark:text-red-400" />
          </div>
        </div>

        {/* Error Content */}
        <div className="space-y-2">
          <h3 className="text-lg font-medium text-red-700 dark:text-red-300">
            {title}
          </h3>
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
