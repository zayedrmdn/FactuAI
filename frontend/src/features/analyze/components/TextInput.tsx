import React from 'react';
import { toast } from 'sonner';
import { franc } from 'franc';
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

      const lang = franc(txt.slice(0, 500));
      if (lang !== 'eng' && lang !== 'und') {
        toast.error('Only English text supported right now.');
        return;
      }

      setInput(txt);
      toast.success('File loaded');
    } catch (error) {
      console.error('Error processing file:', error);
      toast.error('Failed to process file');
    }
  };

  const handlePaste = async (e: React.ClipboardEvent) => {
    e.preventDefault();
    const text = e.clipboardData.getData('text');
    if (!text) return;

    const lang = franc(text.slice(0, 500));
    if (lang !== 'eng' && lang !== 'und') {
      toast.error('Only English text supported');
      return;
    }

    setInput(text);
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
        onPaste={handlePaste}
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
