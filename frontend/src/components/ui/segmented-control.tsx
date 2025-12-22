// Path: frontend/src/components/ui/segmented-control.tsx
'use client';

import * as React from 'react';
import { cn } from '@/lib/utils';

interface SegmentedControlProps {
  value: string;
  onValueChange: (value: string) => void;
  children: React.ReactNode;
  className?: string;
}

interface SegmentOptionProps {
  value: string;
  label?: string;
  icon?: React.ReactNode;
  className?: string;
}

const SegmentedControlContext = React.createContext<{
  value: string;
  onValueChange: (value: string) => void;
} | null>(null);

function SegmentedControl({ value, onValueChange, children, className }: SegmentedControlProps) {
  return (
    <SegmentedControlContext.Provider value={{ value, onValueChange }}>
      <div
        className={cn(
          'relative inline-flex items-center gap-1 p-1 rounded-full bg-muted/50 backdrop-blur-sm border border-border/50',
          className
        )}
        role="radiogroup"
      >
        {children}
      </div>
    </SegmentedControlContext.Provider>
  );
}

function SegmentOption({ value, label, icon, className }: SegmentOptionProps) {
  const context = React.useContext(SegmentedControlContext);
  if (!context) {
    throw new Error('SegmentOption must be used within SegmentedControl');
  }

  const { value: selectedValue, onValueChange } = context;
  const isActive = selectedValue === value;

  return (
    <button
      type="button"
      role="radio"
      aria-checked={isActive}
      onClick={() => onValueChange(value)}
      className={cn(
        'relative px-4 py-2 rounded-full text-sm font-medium transition-all duration-300 ease-out',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
        isActive
          ? 'text-primary-foreground bg-primary shadow-sm scale-105'
          : 'text-muted-foreground hover:text-foreground hover:bg-muted/30',
        className
      )}
    >
      <span className="relative z-10 flex items-center gap-2">
        {icon && (
          <span
            className={cn(
              'transition-transform duration-300',
              isActive ? 'scale-110' : 'scale-100'
            )}
          >
            {icon}
          </span>
        )}
        {label && <span>{label}</span>}
      </span>
    </button>
  );
}

export { SegmentedControl, SegmentOption };
