import React from 'react';

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
      className={`border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-8 text-center transition-colors hover:border-gray-400 dark:hover:border-gray-500 ${
        isProcessing || disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
      } ${className}`}
      onClick={pickFile}
    >
      <Icon className="w-12 h-12 mx-auto text-gray-400 mb-4" />
      <p className="text-sm text-muted-foreground mb-2">{title}</p>
      <p className="text-xs text-muted-foreground mb-4">{description}</p>
      
      <button
        onClick={(e) => {
          e.stopPropagation();
          pickFile();
        }}
        disabled={isProcessing || disabled}
        className="flex items-center gap-2 bg-gray-600 hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed text-white px-4 py-2 rounded text-sm transition mx-auto"
      >
        <Icon className="w-4 h-4" />
        {isProcessing ? "Processing..." : buttonText}
      </button>
    </div>
  );
}
