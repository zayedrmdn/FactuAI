import React from 'react';
import { ArrowPathIcon } from '@heroicons/react/24/outline';

interface ProcessingStatusProps {
  isProcessing: boolean;
  message?: string;
  progress?: number;
  className?: string;
}

export function ProcessingStatus({ 
  isProcessing, 
  message = "Processing...", 
  progress,
  className = "" 
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
