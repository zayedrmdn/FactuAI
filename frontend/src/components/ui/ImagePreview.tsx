'use client';
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

interface ImagePreviewData {
  readonly url: string;
  readonly aiScore: number | null;
  readonly aiError?: string;
  readonly extractedText: string;
}

interface Props {
  readonly imagePreview: ImagePreviewData;
  readonly validationResult: { readonly isValid: boolean };
  readonly input: string;
  readonly loading: null | 'summary' | 'factcheck';
  readonly onRetry: () => void;
  readonly onFactCheck: () => void;
  readonly onClear: () => void;
  readonly openSettings: () => void;
}

/** Helper function to get color based on AI score - using CSS variables */
function getScoreColor(score: number): string {
  if (score >= 70) return 'oklch(var(--score-very-low))'; // Red for high AI detection
  if (score >= 30) return 'oklch(var(--score-medium))'; // Amber for medium
  return 'oklch(var(--score-very-high))'; // Green for likely real
}

/** Helper function to get label based on AI score */
function getScoreLabel(score: number): string {
  if (score >= 70) return 'Likely AI';
  if (score >= 30) return 'Possibly AI';
  return 'Likely Real';
}

/** Sub-component for AI detection error state */
function AIDetectionError({ error }: { readonly error: string }) {
  return (
    <div className="text-center text-sm text-muted-foreground p-4 border rounded-lg">
      <p className="font-medium mb-1">AI Detection</p>
      <p className="text-xs text-red-500">{error}</p>
    </div>
  );
}

/** Sub-component for AI score display */
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

/** Sub-component for AI detection unavailable state */
function AIDetectionUnavailable() {
  return (
    <div className="text-center text-sm text-muted-foreground p-4">AI detection not available</div>
  );
}

/** Sub-component for rendering AI detection section */
function AIDetectionSection({ imagePreview }: { readonly imagePreview: ImagePreviewData }) {
  if (imagePreview.aiError) {
    return <AIDetectionError error={imagePreview.aiError} />;
  }

  if (imagePreview.aiScore !== null) {
    return <AIScoreDisplay score={imagePreview.aiScore} />;
  }

  return <AIDetectionUnavailable />;
}

const FALLBACK_IMAGE =
  'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE2MCIgdmlld0JveD0iMCAwIDIwMCAxNjAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMTYwIiBmaWxsPSIjRjNGNEY2Ii8+Cjx0ZXh0IHg9IjEwMCIgeT0iODAiIGZpbGw9IiM5Q0E0QUYiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIiBmb250LXNpemU9IjE0IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5JbWFnZSBub3QgYXZhaWxhYmxlPC90ZXh0Pgo8L3N2Zz4=';

export default function ImagePreview({
  imagePreview,
  validationResult,
  input,
  loading,
  onRetry,
  onFactCheck,
  onClear,
  openSettings,
}: Readonly<Props>) {
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
