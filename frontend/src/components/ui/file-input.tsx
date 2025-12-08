'use client';

import React from 'react';
import Image from 'next/image';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';
import {
  ArrowPathIcon,
  ShieldCheckIcon,
  PhotoIcon,
  Cog6ToothIcon,
} from '@heroicons/react/24/outline';
import { cn } from '@/lib/utils';

// ========================================================================================
// FILE DROP ZONE COMPONENT
// ========================================================================================

interface FileDropZoneProps {
  readonly onFileSelect: (file: File) => void;
  readonly accept: string;
  readonly isProcessing: boolean;
  readonly icon: React.ComponentType<{ className?: string }>;
  readonly title: string;
  readonly description: string;
  readonly buttonText: string;
  readonly disabled?: boolean;
  readonly className?: string;
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
  className = '',
}: FileDropZoneProps) {
  const pickFile = () => {
    if (isProcessing || disabled) return;

    const input = document.createElement('input');
    input.type = 'file';
    input.accept = accept;
    input.onchange = (e: Event) => {
      const target = e.target as HTMLInputElement;
      const file = target.files?.[0];
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
    <button
      type="button"
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onClick={pickFile}
      disabled={isProcessing || disabled}
      className={cn(
        'group relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-muted-foreground/25 bg-muted/5 px-6 py-10 text-center transition-all duration-200 hover:border-primary/50 hover:bg-muted/20 w-full',
        (isProcessing || disabled) && 'pointer-events-none opacity-60',
        className
      )}
    >
      <div className="mb-4 rounded-full bg-muted p-3 ring-1 ring-border transition-all group-hover:scale-110 group-hover:bg-background">
        <Icon className="h-8 w-8 text-muted-foreground transition-colors group-hover:text-primary" />
      </div>

      <span className="mb-2 text-lg font-semibold text-foreground">{title}</span>
      <span className="mb-6 text-sm text-muted-foreground max-w-xs mx-auto leading-relaxed">
        {description}
      </span>

      <span
        className={cn(
          'inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors',
          'bg-secondary text-secondary-foreground hover:bg-secondary/80',
          'h-10 px-4 py-2'
        )}
      >
        <Icon className="mr-2 h-4 w-4" />
        {isProcessing ? 'Processing...' : buttonText}
      </span>
    </button>
  );
}

// ========================================================================================
// IMAGE PREVIEW COMPONENT
// ========================================================================================

interface ImagePreviewData {
  readonly url: string;
  readonly aiScore: number | null;
  readonly aiError?: string;
  readonly extractedText: string;
}

interface ImagePreviewProps {
  readonly imagePreview: ImagePreviewData;
  readonly validationResult: { readonly isValid: boolean };
  readonly input: string;
  readonly loading: null | 'summary' | 'factcheck';
  readonly onRetry: () => void;
  readonly onFactCheck: () => void;
  readonly onClear: () => void;
  readonly openSettings: () => void;
}

const FALLBACK_IMAGE =
  'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE2MCIgdmlld0JveD0iMCAwIDIwMCAxNjAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMTYwIiBmaWxsPSIjRjNGNEY2Ii8+Cjx0ZXh0IHg9IjEwMCIgeT0iODAiIGZpbGw9IiM5Q0E0QUYiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIiBmb250LXNpemU9IjE0IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5JbWFnZSBub3QgYXZhaWxhYmxlPC90ZXh0Pgo8L3N2Zz4=';

/** Get color based on AI score */
function getScoreColor(score: number): string {
  if (score >= 70) return 'oklch(var(--score-very-low))';
  if (score >= 30) return 'oklch(var(--score-medium))';
  return 'oklch(var(--score-very-high))';
}

/** Get label based on AI score */
function getScoreLabel(score: number): string {
  if (score >= 70) return 'Likely AI';
  if (score >= 30) return 'Possibly AI';
  return 'Likely Real';
}

/** AI detection error state */
function AIDetectionError({ error }: { readonly error: string }) {
  return (
    <div className="text-center text-sm text-muted-foreground p-4 border rounded-lg">
      <p className="font-medium mb-1">AI Detection</p>
      <p className="text-xs text-red-500">{error}</p>
    </div>
  );
}

/** AI score display */
function AIScoreDisplay({ score }: { readonly score: number }) {
  const color = getScoreColor(score);
  const label = getScoreLabel(score);

  return (
    <div className="w-32 h-32">
      <CircularProgressbar
        value={score}
        text={`${score.toFixed(1)}%`}
        styles={buildStyles({
          pathColor: color,
          textColor: color,
          trailColor: 'oklch(var(--score-trail))',
        })}
      />
      <div className="text-center mt-2">
        <div className="text-sm font-medium" style={{ color }}>
          {label}
        </div>
      </div>
    </div>
  );
}

/** AI detection unavailable state */
function AIDetectionUnavailable() {
  return (
    <div className="text-center text-sm text-muted-foreground p-4">AI detection not available</div>
  );
}

/** AI detection section */
function AIDetectionSection({ imagePreview }: { readonly imagePreview: ImagePreviewData }) {
  if (imagePreview.aiError) {
    return <AIDetectionError error={imagePreview.aiError} />;
  }

  if (imagePreview.aiScore !== null) {
    return <AIScoreDisplay score={imagePreview.aiScore} />;
  }

  return <AIDetectionUnavailable />;
}

export function ImagePreview({
  imagePreview,
  validationResult,
  input,
  loading,
  onRetry,
  onFactCheck,
  onClear,
  openSettings,
}: ImagePreviewProps) {
  const hasExtractedText =
    imagePreview.extractedText && imagePreview.extractedText.trim().length > 0;

  const handleImageError = (e: React.SyntheticEvent<HTMLImageElement>) => {
    e.currentTarget.src = FALLBACK_IMAGE;
  };

  return (
    <Card>
      <CardHeader className="flex justify-between items-center">
        <CardTitle>Image Analysis</CardTitle>
        <div className="flex gap-2">
          <button onClick={openSettings} title="Settings" className="p-1 text-muted-foreground">
            <Cog6ToothIcon className="w-5 h-5" />
          </button>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Image Preview */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-muted-foreground">Image Preview</h3>
            <div className="border rounded-lg overflow-hidden relative h-64">
              <Image
                src={imagePreview.url}
                alt="Uploaded content for analysis"
                fill
                className="object-contain bg-gray-50 dark:bg-gray-900"
                onError={handleImageError}
                unoptimized
              />
            </div>
          </div>

          {/* AI Detection Score */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-muted-foreground">AI Detection</h3>
            <div className="flex justify-center">
              <AIDetectionSection imagePreview={imagePreview} />
            </div>
          </div>
        </div>

        {/* Text Extraction Section */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-muted-foreground">Text Extraction</h3>
          {hasExtractedText ? (
            <div className="space-y-3">
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 max-h-32 overflow-y-auto border">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  {imagePreview.extractedText}
                </p>
              </div>
              <div className="flex justify-center">
                <button
                  disabled={!validationResult.isValid || !input.trim() || !!loading}
                  onClick={onFactCheck}
                  className="flex items-center gap-2 bg-purple-600 hover:bg-purple-700 disabled:opacity-50 text-white px-4 py-2 rounded transition"
                >
                  {loading === 'factcheck' ? (
                    <ArrowPathIcon className="w-5 h-5 animate-spin" />
                  ) : (
                    <ShieldCheckIcon className="w-5 h-5" />
                  )}
                  Fact-Check Text
                </button>
              </div>
            </div>
          ) : (
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 border text-center">
              <p className="text-sm text-muted-foreground">No text found in this image</p>
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3 justify-center">
          <button
            onClick={onRetry}
            className="flex items-center gap-2 bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded transition"
          >
            <PhotoIcon className="w-5 h-5" />
            Try Another Image
          </button>

          <button
            onClick={onClear}
            className="flex items-center gap-2 text-gray-600 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200 px-4 py-2 rounded transition"
          >
            Clear All
          </button>
        </div>
      </CardContent>
    </Card>
  );
}

// Default export for backwards compatibility
export default ImagePreview;
