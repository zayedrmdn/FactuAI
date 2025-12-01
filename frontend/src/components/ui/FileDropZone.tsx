import React from 'react';
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface FileDropZoneProps {
  onFileSelect: (file: File) => void;
  accept: string;
  isProcessing: boolean;
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
  buttonText: string;
  disabled?: boolean;
  className?: string;
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
  className = ""
}: FileDropZoneProps) {
  const pickFile = () => {
    if (isProcessing || disabled) return;
    
    const input = document.createElement("input");
    input.type = "file";
    input.accept = accept;
    input.onchange = (e: any) => {
      const file = e.target.files?.[0];
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
    <div
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onClick={pickFile}
      className={cn(
        "group relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-muted-foreground/25 bg-muted/5 px-6 py-10 text-center transition-all duration-200 hover:border-primary/50 hover:bg-muted/20",
        (isProcessing || disabled) && "pointer-events-none opacity-60",
        className
      )}
    >
      <div className="mb-4 rounded-full bg-muted p-3 ring-1 ring-border transition-all group-hover:scale-110 group-hover:bg-background">
        <Icon className="h-8 w-8 text-muted-foreground transition-colors group-hover:text-primary" />
      </div>
      
      <h3 className="mb-2 text-lg font-semibold text-foreground">{title}</h3>
      <p className="mb-6 text-sm text-muted-foreground max-w-xs mx-auto leading-relaxed">
        {description}
      </p>
      
      <Button
        onClick={(e) => {
          e.stopPropagation();
          pickFile();
        }}
        disabled={isProcessing || disabled}
        variant="secondary"
        className="pointer-events-none" 
      >
        <Icon className="mr-2 h-4 w-4" />
        {isProcessing ? "Processing..." : buttonText}
      </Button>
    </div>
  );
}
