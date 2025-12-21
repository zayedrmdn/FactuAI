'use client';

import React from 'react';
import { cn } from '@/lib/utils';

// ========================================================================================
// FILE DROP ZONE COMPONENT
// ========================================================================================

interface FileDropZoneProps {
  readonly onFileSelect: (file: File) => void;
  readonly accept: string;
  readonly isProcessing: boolean;
  readonly icon: React.ComponentType<{ className?: string }>;
  readonly title: string;
  readonly description: string;
  readonly buttonText: string;
  readonly disabled?: boolean;
  readonly className?: string;
}

export function FileDropZone({
  onFileSelect,
  accept,
  isProcessing,
  icon: Icon,
  title,
  description,
  buttonText,
  disabled = false,
  className = '',
}: FileDropZoneProps) {
  const pickFile = () => {
    if (isProcessing || disabled) return;

    const input = document.createElement('input');
    input.type = 'file';
    input.accept = accept;
    input.onchange = (e: Event) => {
      const target = e.target as HTMLInputElement;
      const file = target.files?.[0];
      if (file) onFileSelect(file);
    };
    input.click();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (isProcessing || disabled) return;

    const file = e.dataTransfer.files?.[0];
    if (file) onFileSelect(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  return (
    <button
      type="button"
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onClick={pickFile}
      disabled={isProcessing || disabled}
      className={cn(
        'group relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-muted-foreground/25 bg-muted/5 px-6 py-10 text-center transition-all duration-200 hover:border-primary/50 hover:bg-muted/20 w-full',
        (isProcessing || disabled) && 'pointer-events-none opacity-60',
        className
      )}
    >
      <div className="mb-4 rounded-full bg-muted p-3 ring-1 ring-border transition-all group-hover:scale-110 group-hover:bg-background">
        <Icon className="h-8 w-8 text-muted-foreground transition-colors group-hover:text-primary" />
      </div>

      <span className="mb-2 text-lg font-semibold text-foreground">{title}</span>
      <span className="mb-6 text-sm text-muted-foreground max-w-xs mx-auto leading-relaxed">
        {description}
      </span>

      <span
        className={cn(
          'inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors',
          'bg-secondary text-secondary-foreground hover:bg-secondary/80',
          'h-10 px-4 py-2'
        )}
      >
        <Icon className="mr-2 h-4 w-4" />
        {isProcessing ? 'Processing...' : buttonText}
      </span>
    </button>
  );
}
