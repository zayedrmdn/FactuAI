'use client';

import * as React from 'react';
import { ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';

interface AccordionItemProps {
  readonly title: React.ReactNode;
  readonly children: React.ReactNode;
  readonly defaultOpen?: boolean;
  readonly className?: string;
}

export function AccordionItem({
  title,
  children,
  defaultOpen = false,
  className,
}: Readonly<AccordionItemProps>) {
  const [isOpen, setIsOpen] = React.useState(defaultOpen);
  const contentRef = React.useRef<HTMLDivElement>(null);

  return (
    <div className={cn('border-b last:border-0', className)}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full items-center justify-between py-4 px-4 text-sm font-medium transition-colors hover:bg-slate-50 text-left"
        aria-expanded={isOpen}
      >
        {title}
        <ChevronDown
          className={cn(
            'h-4 w-4 shrink-0 transition-transform duration-300 ease-in-out',
            isOpen && 'rotate-180'
          )}
        />
      </button>
      <div
        ref={contentRef}
        className={cn(
          'transition-all duration-300 ease-in-out overflow-hidden',
          isOpen ? 'max-h-[2000px] opacity-100' : 'max-h-0 opacity-0'
        )}
      >
        <div className="pb-4 text-sm text-muted-foreground">{children}</div>
      </div>
    </div>
  );
}

export function Accordion({
  children,
  className,
}: Readonly<{
  children: React.ReactNode;
  className?: string;
}>) {
  return <div className={cn('w-full', className)}>{children}</div>;
}
