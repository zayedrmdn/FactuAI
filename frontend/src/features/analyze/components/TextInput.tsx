import React from 'react';
import { toast } from 'sonner';
import { fileToText } from '@/lib/dashboard/fileToText';
import { TextSize } from '@/types/dashboard/ui';
import { cn } from '@/lib/utils';

interface TextTabProps {
  input: string;
  setInput: (value: string) => void;
  textSize: TextSize;
  onClear: () => void;
}

export default function TextInput({ input, setInput, textSize }: Readonly<TextTabProps>) {
  const handleFile = async (file: File) => {
    try {
      const txt = await fileToText(file);
      if (!txt) return;

      setInput(txt);
      toast.success('File loaded');
    } catch (error) {
      console.error('Error processing file:', error);
      toast.error('Failed to process file');
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const textSizeClass = {
    sm: 'text-base',
    md: 'text-lg',
    lg: 'text-xl',
  }[textSize];

  return (
    <div className="relative">
      <textarea
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        placeholder="Paste claim text, social media link, or drag and drop media files here..."
        className={cn(
          'w-full min-h-[180px] p-6 resize-none',
          'bg-transparent text-foreground placeholder:text-muted-foreground/50',
          'focus-visible:outline-none',
          'leading-relaxed',
          textSizeClass
        )}
      />
    </div>
  );
}
