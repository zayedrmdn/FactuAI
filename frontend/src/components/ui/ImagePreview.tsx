"use client";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { CircularProgressbar, buildStyles } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";
import { ArrowPathIcon, ShieldCheckIcon, PhotoIcon, Cog6ToothIcon } from "@heroicons/react/24/outline";

interface ImagePreviewData {
  url: string;
  aiScore: number | null;
  aiError?: string;
  extractedText: string;
}

interface Props {
  imagePreview: ImagePreviewData;
  validationResult: { isValid: boolean };
  input: string;
  loading: null | "summary" | "factcheck";
  onRetry: () => void;
  onFactCheck: () => void;
  onClear: () => void;
  openSettings: () => void;
}

export default function ImagePreview({
  imagePreview,
  validationResult,
  input,
  loading,
  onRetry,
  onFactCheck,
  onClear,
  openSettings,
}: Props) {
  const hasExtractedText = imagePreview.extractedText && imagePreview.extractedText.trim().length > 0;

  return (
    <Card>
      <CardHeader className="flex justify-between items-center">
        <CardTitle>Image Analysis</CardTitle>
        <div className="flex gap-2">
          <button
            onClick={openSettings}
            title="Settings"
            className="p-1 text-muted-foreground"
          >
            <Cog6ToothIcon className="w-5 h-5" />
          </button>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Image Preview */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-muted-foreground">Image Preview</h3>
            <div className="border rounded-lg overflow-hidden">
              <img
                src={imagePreview.url}
                alt="Processed image"
                className="w-full h-64 object-contain bg-gray-50 dark:bg-gray-900"
                onError={(e) => {
                  e.currentTarget.src = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE2MCIgdmlld0JveD0iMCAwIDIwMCAxNjAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMTYwIiBmaWxsPSIjRjNGNEY2Ii8+Cjx0ZXh0IHg9IjEwMCIgeT0iODAiIGZpbGw9IiM5Q0E0QUYiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIiBmb250LXNpemU9IjE0IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5JbWFnZSBub3QgYXZhaWxhYmxlPC90ZXh0Pgo8L3N2Zz4=";
                }}
              />
            </div>
          </div>

          {/* AI Detection Score */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-muted-foreground">AI Detection</h3>
            <div className="flex justify-center">
              {imagePreview.aiError ? (
                <div className="text-center text-sm text-muted-foreground p-4 border rounded-lg">
                  <p className="font-medium mb-1">AI Detection</p>
                  <p className="text-xs text-red-500">{imagePreview.aiError}</p>
                </div>
              ) : imagePreview.aiScore !== null ? (
                <div className="w-32 h-32">
                  <CircularProgressbar
                    value={imagePreview.aiScore}
                    text={`${imagePreview.aiScore.toFixed(1)}%`}
                    styles={buildStyles({
                      pathColor: imagePreview.aiScore >= 70 ? "#dc2626" : 
                               imagePreview.aiScore >= 30 ? "#d97706" : "#16a34a",
                      textColor: imagePreview.aiScore >= 70 ? "#dc2626" : 
                               imagePreview.aiScore >= 30 ? "#d97706" : "#16a34a",
                      trailColor: "#e5e7eb",
                    })}
                  />
                  <div className="text-center mt-2">
                    <div className="text-sm font-medium" style={{ 
                      color: imagePreview.aiScore >= 70 ? "#dc2626" : 
                             imagePreview.aiScore >= 30 ? "#d97706" : "#16a34a" 
                    }}>
                      {imagePreview.aiScore >= 70 ? "Likely AI" :
                       imagePreview.aiScore >= 30 ? "Possibly AI" : "Likely Real"}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center text-sm text-muted-foreground p-4">
                  AI detection not available
                </div>
              )}
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
                  {loading === "factcheck" ? (
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
              <p className="text-sm text-muted-foreground">
                No text found in this image
              </p>
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
