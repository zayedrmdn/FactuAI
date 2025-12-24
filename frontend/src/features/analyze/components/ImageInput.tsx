import React, { useState } from 'react';
import { Image as ImageIcon, Link2, X } from 'lucide-react';
import { useImageProcessing } from '@/lib/hooks/useImageProcessing';
import { FileDropZone } from '@/components/ui/file-input';
import { ProcessingStatus } from '@/components/ui/feedback-states';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface ImagePreviewData {
  imageUrl: string;
  extractedText: string;
  aiScore?: number | undefined;
}

interface ImageTabProps {
  onImageProcessed: (
    text: string,
    aiScore: number | null,
    imageUrl: string,
    aiError?: string
  ) => void;
}

export default function ImageInput({ onImageProcessed }: Readonly<ImageTabProps>) {
  const [imagePreview, setImagePreview] = useState<ImagePreviewData | null>(null);
  const [imageUrl, setImageUrl] = useState('');
  const [imageUrlLoading, setImageUrlLoading] = useState(false);

  const { uploadImage, isProcessing } = useImageProcessing({
    onImageProcessed: (
      text: string,
      aiScore: number | null,
      imageUrl: string,
      aiError?: string
    ) => {
      setImagePreview({
        imageUrl,
        extractedText: text,
        aiScore: aiScore ?? undefined,
      });
      onImageProcessed(text, aiScore, imageUrl, aiError);
    },
  });

  const handleImageUrl = async () => {
    if (!imageUrl.trim()) return;

    setImageUrlLoading(true);
    try {
      const response = await fetch('/api/extract-text-from-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: imageUrl }),
      });

      if (!response.ok) throw new Error('Failed to extract text from URL');

      const data = await response.json();
      setImagePreview({
        imageUrl,
        extractedText: data.text || '',
        aiScore: data.ai_percentage ?? undefined,
      });
      onImageProcessed(data.text || '', data.ai_percentage || null, imageUrl, data.ai_error);
    } catch (error) {
      console.error('URL processing error:', error);
    } finally {
      setImageUrlLoading(false);
    }
  };

  const handleClearPreview = () => {
    if (imagePreview?.imageUrl.startsWith('blob:')) {
      URL.revokeObjectURL(imagePreview.imageUrl);
    }
    setImagePreview(null);
    setImageUrl('');
  };

  return (
    <div className="space-y-6">
      {imagePreview ? (
        <div className="bg-card/50 border border-border/50 rounded-xl overflow-hidden animate-in fade-in zoom-in-95 duration-300">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-border/50 bg-muted/20">
            <h3 className="font-semibold text-sm flex items-center gap-2">
              <ImageIcon className="h-4 w-4 text-primary" />
              Image Analyzed
            </h3>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClearPreview}
              className="h-7 px-2 text-muted-foreground hover:text-destructive hover:bg-destructive/10 -mr-1"
            >
              <X className="h-3.5 w-3.5 mr-1.5" />
              Clear
            </Button>
          </div>

          <div className="p-4 sm:p-6 grid gap-6 md:grid-cols-2">
            {/* Image Preview */}
            <div className="relative group rounded-lg overflow-hidden bg-black/5 dark:bg-black/20 border border-border/50">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={imagePreview.imageUrl}
                alt="Analyzed content"
                className="w-full h-48 sm:h-64 object-contain"
              />

              {imagePreview.aiScore !== undefined && (
                <div className="absolute top-2 left-2 px-2.5 py-1 rounded-full bg-background/90 backdrop-blur text-xs font-semibold shadow-sm border border-border/50">
                  <span
                    className={cn(
                      'mr-1.5 inline-block w-2 h-2 rounded-full',
                      imagePreview.aiScore > 50 ? 'bg-destructive' : 'bg-success'
                    )}
                  />
                  {imagePreview.aiScore.toFixed(0)}% AI Probability
                </div>
              )}
            </div>

            {/* Extracted Text */}
            <div className="space-y-2">
              <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Extracted Text
              </label>
              <div className="relative">
                <div className="p-3 rounded-lg bg-muted/30 border border-border/50 text-sm leading-relaxed max-h-48 sm:max-h-64 overflow-y-auto">
                  {imagePreview.extractedText ? (
                    <p className="whitespace-pre-wrap">{imagePreview.extractedText}</p>
                  ) : (
                    <p className="text-muted-foreground italic">No readable text found in image.</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          <FileDropZone
            onFileSelect={uploadImage}
            accept="image/*"
            isProcessing={isProcessing}
            icon={ImageIcon}
            title="Upload Image"
            description="Drag & drop or click to upload. Supports JPG, PNG, WEBP."
            buttonText="Select Image"
          />

          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-border/50" />
            </div>
            <div className="relative flex justify-center text-xs uppercase tracking-wider font-semibold text-muted-foreground/60">
              <span className="bg-card px-2">Or paste URL</span>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-2 sm:items-center">
            <div className="relative flex-1 min-w-0 group">
              <div className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground/50 group-focus-within:text-primary transition-colors">
                <Link2 className="h-4 w-4" />
              </div>
              <input
                type="url"
                value={imageUrl}
                onChange={(e) => setImageUrl(e.target.value)}
                placeholder="https://example.com/image.jpg"
                className="w-full h-10 pl-9 pr-3 rounded-lg border border-border bg-muted/20 text-sm placeholder:text-muted-foreground/50 focus:bg-background focus:border-primary/30 focus:ring-2 focus:ring-primary/10 transition-all outline-none"
                disabled={isProcessing || imageUrlLoading}
              />
            </div>
            <Button
              onClick={handleImageUrl}
              disabled={!imageUrl.trim() || isProcessing || imageUrlLoading}
              className="h-10 px-6 font-medium sm:w-auto w-full"
            >
              {imageUrlLoading ? 'Loading...' : 'Extract'}
            </Button>
          </div>

          <ProcessingStatus
            isProcessing={isProcessing || imageUrlLoading}
            message="Analyzing image..."
          />
        </div>
      )}
    </div>
  );
}
