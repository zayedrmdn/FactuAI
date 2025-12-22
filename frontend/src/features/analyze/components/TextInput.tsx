import React from 'react';
import { Clipboard, Upload, X } from 'lucide-react';
import { toast } from 'sonner';
import { franc } from 'franc';
import { fileToText } from '@/lib/dashboard/fileToText';
import { TextSize } from '@/types/dashboard/ui';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';

interface PasteUploadClearProps {
  onPaste: () => void;
  onUpload: () => void;
  onClear: () => void;
  disabled?: boolean;
}

function PasteUploadClear({
  onPaste,
  onUpload,
  onClear,
  disabled,
}: Readonly<PasteUploadClearProps>) {
  return (
    <div className="flex gap-2">
      <Button
        variant="outline"
        size="sm"
        onClick={onPaste}
        disabled={disabled}
        className="h-7 text-xs font-medium text-muted-foreground hover:text-foreground bg-background hover:bg-muted/50 border-input shadow-sm transition-all duration-200"
      >
        <Clipboard className="h-3 w-3 mr-1.5" />
        Paste
      </Button>
      <Button
        variant="outline"
        size="sm"
        onClick={onUpload}
        disabled={disabled}
        className="h-7 text-xs font-medium text-muted-foreground hover:text-foreground bg-background hover:bg-muted/50 border-input shadow-sm transition-all duration-200"
      >
        <Upload className="h-3 w-3 mr-1.5" />
        Upload
      </Button>
      <div className="h-4 w-px bg-border my-auto mx-1" /> {/* Divider for visual separation */}
      <Button
        variant="ghost"
        size="sm"
        onClick={onClear}
        disabled={disabled}
        className="h-7 text-xs text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
      >
        <X className="h-3 w-3 mr-1.5" />
        Clear
      </Button>
    </div>
  );
}

interface TextTabProps {
  input: string;
  setInput: (value: string) => void;
  textSize: TextSize;
  onClear: () => void;
}

export default function TextInput({ input, setInput, textSize, onClear }: Readonly<TextTabProps>) {
  const wordCount = input.trim().split(/\s+/).filter(Boolean).length;

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

  const pickFile = () => {
    const inputEl = document.createElement('input');
    inputEl.type = 'file';
    inputEl.accept = '.txt,.pdf';
    inputEl.onchange = (e: Event) => {
      const target = e.target as HTMLInputElement;
      const file = target.files?.[0];
      if (file) handleFile(file);
    };
    inputEl.click();
  };

  const pasteFromClipboard = async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (!text) {
        toast.error('Clipboard is empty');
        return;
      }

      const lang = franc(text.slice(0, 500));
      if (lang !== 'eng' && lang !== 'und') {
        toast.error('Only English text supported');
        return;
      }

      setInput(text);
      toast.success('Text pasted from clipboard');
    } catch (error) {
      console.error('Error reading clipboard:', error);
      toast.error('Failed to read clipboard');
    }
  };

  const textSizeClass = {
    sm: 'text-base',
    md: 'text-lg',
    lg: 'text-xl',
  }[textSize];

  return (
    <div className="space-y-2 relative group">
      <div className="flex items-center justify-between px-1">
        <PasteUploadClear
          onPaste={pasteFromClipboard}
          onUpload={pickFile}
          onClear={onClear}
          disabled={false}
        />
        <div className="text-xs font-mono text-muted-foreground/60">{wordCount} words</div>
      </div>

      <div className="relative">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Paste claim or text to analyze..."
          className={cn(
            'w-full h-64 p-4 rounded-xl resize-none bg-muted/30 focus:bg-background transition-colors duration-200',
            'border-2 border-transparent focus:border-primary/10',
            'text-foreground placeholder:text-muted-foreground/50',
            'focus-visible:outline-none',
            'leading-relaxed',
            textSizeClass
          )}
        />
      </div>
    </div>
  );
}
