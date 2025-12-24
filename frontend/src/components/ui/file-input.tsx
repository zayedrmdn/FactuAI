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
        'group relative flex flex-col items-center justify-center rounded-xl border border-dashed border-border bg-card/50 px-8 py-12 text-center transition-all duration-300 hover:border-primary/50 hover:bg-muted/30 hover:shadow-sm w-full outline-none focus-visible:ring-2 focus-visible:ring-primary/20',
        (isProcessing || disabled) && 'pointer-events-none opacity-60',
        className
      )}
    >
      <div className="mb-5 rounded-2xl bg-muted/50 p-4 ring-1 ring-border/50 transition-all duration-300 group-hover:scale-105 group-hover:bg-background group-hover:ring-primary/20 group-hover:shadow-sm">
        <Icon className="h-8 w-8 text-muted-foreground transition-colors group-hover:text-primary" />
      </div>

      <div className="space-y-1 mb-6">
        <span className="block text-lg font-semibold text-foreground tracking-tight group-hover:text-primary transition-colors">
          {title}
        </span>
        <span className="block text-sm text-muted-foreground/80 max-w-xs mx-auto leading-relaxed">
          {description}
        </span>
      </div>

      <span
        className={cn(
          'inline-flex items-center justify-center whitespace-nowrap rounded-lg text-sm font-medium transition-all',
          'bg-primary/10 text-primary border border-primary/10 group-hover:bg-primary group-hover:text-primary-foreground group-hover:border-primary/20',
          'h-10 px-6 shadow-sm'
        )}
      >
        <Icon className="mr-2 h-4 w-4" />
        {isProcessing ? 'Processing content...' : buttonText}
      </span>
    </button>
  );
}
