import React, { useState } from 'react';
import { PhotoIcon } from '@heroicons/react/24/outline';
import { useImageProcessing } from '../../../hooks/useImageProcessing';
import { FileDropZone } from '@/components/ui/FileDropZone';
import { ProcessingStatus } from '@/components/ui/ProcessingStatus';

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

export default function ImageTab({ onImageProcessed }: Readonly<ImageTabProps>) {
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
              className="text-xs text-gray-500 hover:text-gray-700"
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
                <span className={imagePreview.aiScore > 50 ? 'text-red-600' : 'text-green-600'}>
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
            icon={PhotoIcon}
            title="Drop an image here or click to upload"
            description="Text will be extracted using OCR technology"
            buttonText="Select Image File"
          />

          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-300" />
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
              className="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm"
              disabled={isProcessing || imageUrlLoading}
            />
            <button
              onClick={handleImageUrl}
              disabled={!imageUrl.trim() || isProcessing || imageUrlLoading}
              className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700 disabled:opacity-50"
            >
              {imageUrlLoading ? 'Loading...' : 'Extract'}
            </button>
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
