import React, { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { ShieldCheck, Settings, Loader2 } from 'lucide-react';
import { validateBasic } from '@/lib/dashboard/validation';
import InputTabs from './InputTabs';
import { AnalysisModeToggle } from './AnalysisModeToggle';
import { InputType, TextSize } from '@/types/dashboard/ui';

interface InputCardProps {
  input: string;
  setInput: (v: string) => void;
  loading: null | 'summary' | 'factcheck';
  onFactCheck: () => void;
  onClear: () => void;
  textSize: TextSize;
  openSettings: () => void;
  onAIDetection?: (score: number | null, error?: string) => void;
  onInputTypeChange?: (
    type: InputType,
    data?: {
      imageData?: { url: string; aiScore: number | null; aiError?: string | undefined };
      videoData?: { filename: string; videoUrl?: string | undefined };
    }
  ) => void;
  saveImageToHistory?: (
    text: string,
    imageUrl: string,
    aiScore: number | null,
    aiError?: string
  ) => void;
  saveVideoToHistory?: (text: string, filename: string, videoUrl?: string) => void;
}

export default function InputCard({
  input,
  setInput,
  loading,
  onFactCheck,
  onClear,
  textSize,
  openSettings,
  onAIDetection,
  onInputTypeChange,
  saveImageToHistory,
  saveVideoToHistory,
}: InputCardProps) {
  const [, setCurrentInputType] = useState<InputType>('text');

  const validationResult = validateBasic(input);
  const showValidationError = input.trim().length > 0 && validationResult.error;

  const handleImageProcessed = useCallback(
    (text: string, aiScore: number | null, imageUrl: string, aiError?: string) => {
      setInput(text);
      setCurrentInputType('image');

      const imageData = { url: imageUrl, aiScore, aiError: aiError ?? undefined };

      if (onAIDetection) {
        onAIDetection(aiScore, aiError);
      }

      if (onInputTypeChange) {
        onInputTypeChange('image', { imageData });
      }

      if (saveImageToHistory) {
        saveImageToHistory(text, imageUrl, aiScore, aiError);
      }
    },
    [setInput, onAIDetection, onInputTypeChange, saveImageToHistory]
  );

  const handleVideoProcessed = useCallback(
    (text: string, filename?: string, videoUrl?: string) => {
      setInput(text);
      setCurrentInputType('video');

      const videoData = {
        filename: filename || 'Unknown video',
        videoUrl: videoUrl,
      };

      if (onInputTypeChange) {
        onInputTypeChange('video', { videoData });
      }

      if (saveVideoToHistory && text.trim() && filename) {
        saveVideoToHistory(text, filename, videoUrl);
      }
    },
    [setInput, onInputTypeChange, saveVideoToHistory]
  );

  const handleTextInput = useCallback(
    (text: string) => {
      setInput(text);
      setCurrentInputType('text');

      if (onInputTypeChange) {
        onInputTypeChange('text');
      }
    },
    [setInput, onInputTypeChange]
  );

  const handleInputTypeChange = useCallback((type: InputType) => {
    setCurrentInputType(type);
  }, []);

  const handleClear = useCallback(() => {
    setInput('');
    setCurrentInputType('text');
    onClear();
  }, [setInput, onClear]);

  return (
    <div className="w-full flex flex-col gap-4 animate-in fade-in duration-500">
      {/* Glowing Input Container */}
      <div className="relative group">
        {/* Glow effect behind input */}
        <div className="absolute -inset-0.5 bg-gradient-to-r from-primary via-primary/50 to-primary rounded-xl opacity-0 group-hover:opacity-20 group-focus-within:opacity-30 transition-opacity duration-500 blur-md" />

        {/* Main Input Card */}
        <div className="relative bg-card rounded-xl border border-border overflow-hidden shadow-sm group-hover:shadow-lg group-focus-within:shadow-lg group-focus-within:border-primary/30 transition-all duration-300">
          {/* Settings Button - Top Right */}
          <div className="absolute top-3 right-3 z-10">
            <Button
              variant="ghost"
              size="sm"
              onClick={openSettings}
              className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground hover:bg-muted rounded-lg"
              aria-label="Settings"
            >
              <Settings className="h-4 w-4" />
            </Button>
          </div>

          {/* Input Tabs Content */}
          <InputTabs
            input={input}
            setInput={handleTextInput}
            textSize={textSize}
            onClear={handleClear}
            onImageProcessed={handleImageProcessed}
            onVideoProcessed={handleVideoProcessed}
            onInputTypeChange={handleInputTypeChange}
          />

          {/* Bottom Action Bar */}
          {input && (
            <div className="px-4 pb-4 pt-2 border-t border-border/50 bg-muted/30 animate-in slide-in-from-bottom-2 duration-200">
              <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4">
                {/* Analysis Mode */}
                <div className="flex items-center gap-3">
                  <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                    Mode
                  </span>
                  <AnalysisModeToggle />
                </div>

                {/* Action Buttons */}
                <div className="flex items-center gap-2">
                  <Button
                    variant="ghost"
                    onClick={handleClear}
                    disabled={loading !== null}
                    size="sm"
                    className="text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                  >
                    Clear
                  </Button>
                  <Button
                    onClick={onFactCheck}
                    disabled={loading !== null || !validationResult.isValid}
                    className="gap-2 px-6 font-semibold shadow-md shadow-primary/20 hover:shadow-lg hover:shadow-primary/30 transition-all"
                    size="sm"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        {loading === 'summary' ? 'Reading...' : 'Verifying...'}
                      </>
                    ) : (
                      <>
                        <ShieldCheck className="h-4 w-4" />
                        Analyze
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Validation Error */}
      {showValidationError && (
        <div className="rounded-lg bg-destructive/5 border border-destructive/10 p-4 text-sm animate-in fade-in slide-in-from-top-1">
          <div className="flex items-start gap-2">
            <div className="mt-0.5">
              <div className="h-1.5 w-1.5 rounded-full bg-destructive" />
            </div>
            <div>
              <p className="font-medium text-destructive">{validationResult.error}</p>
              {validationResult.suggestion && (
                <p className="mt-1 text-xs opacity-80 text-destructive/80">
                  {validationResult.suggestion}
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
