import React, { useState } from 'react';
import { Image as ImageIcon } from 'lucide-react';
import { useImageProcessing } from '@/lib/hooks/useImageProcessing';
import { FileDropZone } from '@/components/ui/file-input';
import { ProcessingStatus } from '@/components/ui/feedback-states';
import { Button } from '@/components/ui/button';

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
    <div className="space-y-4">
      {imagePreview ? (
        <div className="border rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-medium text-sm">Image Processed</h3>
            <button
              onClick={handleClearPreview}
              className="text-xs text-muted-foreground hover:text-foreground"
            >
              Clear
            </button>
          </div>

          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={imagePreview.imageUrl}
            alt="Uploaded content preview"
            className="w-full max-h-40 object-contain rounded"
          />

          <div className="space-y-2">
            {imagePreview.aiScore !== undefined && (
              <div className="text-xs">
                <span className="font-medium">AI Detection: </span>
                <span className={imagePreview.aiScore > 50 ? 'text-destructive' : 'text-success'}>
                  {imagePreview.aiScore.toFixed(1)}% AI-generated
                </span>
              </div>
            )}

            <div className="text-xs text-muted-foreground">
              <p className="font-medium mb-1">Extracted Text:</p>
              <p className="line-clamp-3">{imagePreview.extractedText || 'No text found'}</p>
            </div>
          </div>
        </div>
      ) : (
        <>
          <FileDropZone
            onFileSelect={uploadImage}
            accept="image/*"
            isProcessing={isProcessing}
            icon={ImageIcon}
            title="Drop an image here or click to upload"
            description="Text will be extracted using OCR technology"
            buttonText="Select Image File"
          />

          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-border" />
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-background text-muted-foreground">or</span>
            </div>
          </div>

          <div className="flex gap-2">
            <input
              type="url"
              value={imageUrl}
              onChange={(e) => setImageUrl(e.target.value)}
              placeholder="Enter image URL..."
              className="flex-1 px-3 py-2 border border-input rounded-md text-sm bg-background text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              disabled={isProcessing || imageUrlLoading}
            />
            <Button
              onClick={handleImageUrl}
              type="button"
              disabled={!imageUrl.trim() || isProcessing || imageUrlLoading}
            >
              {imageUrlLoading ? 'Loading...' : 'Extract'}
            </Button>
          </div>

          <ProcessingStatus
            isProcessing={isProcessing || imageUrlLoading}
            message="Extracting text from image..."
          />
        </>
      )}
    </div>
  );
}
