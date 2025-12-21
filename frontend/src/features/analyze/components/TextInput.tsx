import React from 'react';
import { Clipboard, Upload, X } from 'lucide-react';
import { toast } from 'sonner';
import { franc } from 'franc';
import { fileToText } from '@/lib/dashboard/fileToText';
import { validateBasic } from '@/lib/dashboard/validation';
import { TextSize } from '@/types/dashboard/ui';

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
      <button
        onClick={onPaste}
        disabled={disabled}
        className="flex items-center gap-1 px-3 py-1 text-xs border border-input rounded hover:bg-accent hover:text-accent-foreground disabled:opacity-50 transition-colors"
      >
        <Clipboard className="h-3 w-3" />
        Paste
      </button>
      <button
        onClick={onUpload}
        disabled={disabled}
        className="flex items-center gap-1 px-3 py-1 text-xs border border-input rounded hover:bg-accent hover:text-accent-foreground disabled:opacity-50 transition-colors"
      >
        <Upload className="h-3 w-3" />
        Upload
      </button>
      <button
        onClick={onClear}
        disabled={disabled}
        className="flex items-center gap-1 px-3 py-1 text-xs border border-input rounded hover:bg-accent hover:text-accent-foreground disabled:opacity-50 transition-colors"
      >
        <X className="h-3 w-3" />
        Clear
      </button>
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
  const validationResult = validateBasic(input);
  const wordCount = input.trim().split(/\s+/).filter(Boolean).length;
  const showValidationError = input.trim().length > 0 && validationResult.error;

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
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg',
  }[textSize];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <PasteUploadClear
          onPaste={pasteFromClipboard}
          onUpload={pickFile}
          onClear={onClear}
          disabled={validationResult.isValid === false}
        />
        <div className="text-xs text-muted-foreground">{wordCount} words</div>
      </div>

      <textarea
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Enter text to fact-check, or use the buttons above to paste/upload..."
        className={`w-full h-64 p-4 border border-input rounded-lg resize-none bg-background text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ${textSizeClass}`}
      />

      {showValidationError && (
        <div className="text-sm text-destructive">
          <p className="font-medium">{validationResult.error}</p>
          {validationResult.suggestion && (
            <p className="text-xs text-muted-foreground mt-1">{validationResult.suggestion}</p>
          )}
        </div>
      )}
    </div>
  );
}
