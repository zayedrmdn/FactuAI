/**
 * UI Primitives - Simple, reusable UI components
 * Includes: Badge, Separator, Progress
 */

'use client';

import * as React from 'react';
import * as SeparatorPrimitive from '@radix-ui/react-separator';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

// ========================================================================================
// BADGE COMPONENT
// ========================================================================================

const badgeVariants = cva(
  'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
  {
    variants: {
      variant: {
        default: 'border-transparent bg-primary text-primary-foreground hover:bg-primary/80',
        secondary:
          'border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80',
        destructive:
          'border-transparent bg-destructive text-destructive-foreground hover:bg-destructive/80',
        outline: 'text-foreground',
        success: 'border-transparent bg-success text-success-foreground hover:bg-success/90',
        warning: 'border-transparent bg-warning text-warning-foreground hover:bg-warning/90',
        info: 'border-transparent bg-info text-info-foreground hover:bg-info/90',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>, VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

// ========================================================================================
// SEPARATOR COMPONENT
// ========================================================================================

const Separator = React.forwardRef<
  React.ElementRef<typeof SeparatorPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SeparatorPrimitive.Root>
>(({ className, orientation = 'horizontal', decorative = true, ...props }, ref) => (
  <SeparatorPrimitive.Root
    ref={ref}
    decorative={decorative}
    orientation={orientation}
    className={cn(
      'shrink-0 bg-border',
      orientation === 'horizontal' ? 'h-[1px] w-full' : 'h-full w-[1px]',
      className
    )}
    {...props}
  />
));
Separator.displayName = SeparatorPrimitive.Root.displayName;

// ========================================================================================
// PROGRESS COMPONENT
// ========================================================================================

interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value?: number;
  variant?: 'default' | 'success' | 'warning' | 'destructive' | 'info';
}

const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(
  ({ className, value, variant = 'default', ...props }, ref) => (
    <div
      ref={ref}
      className={cn('relative h-2 w-full overflow-hidden rounded-full bg-secondary', className)}
      style={
        {
          '--progress': `${Math.max(0, Math.min(100, value ?? 0))}%`,
        } as React.CSSProperties
      }
      {...props}
    >
      <div
        className={cn(
          'h-full w-[var(--progress)] transition-[width] duration-300 ease-out',
          variant === 'success'
            ? 'bg-success'
            : variant === 'warning'
              ? 'bg-warning'
              : variant === 'destructive'
                ? 'bg-destructive'
                : variant === 'info'
                  ? 'bg-info'
                  : 'bg-primary'
        )}
      />
    </div>
  )
);
Progress.displayName = 'Progress';

// ========================================================================================
// EXPORTS
// ========================================================================================

export { Badge, badgeVariants, Separator, Progress };
