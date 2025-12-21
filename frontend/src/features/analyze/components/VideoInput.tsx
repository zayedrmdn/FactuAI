import React, { useState } from 'react';
import { Video } from 'lucide-react';
import { useVideoProcessing } from '@/lib/hooks/useVideoProcessing';
import { FileDropZone } from '@/components/ui/file-input';
import { ProcessingStatus } from '@/components/ui/feedback-states';

interface VideoPreviewData {
  filename: string;
  extractedText: string;
  videoUrl: string;
}

interface VideoTabProps {
  onVideoProcessed: (text: string, filename?: string, videoUrl?: string) => void;
}

export default function VideoInput({ onVideoProcessed }: Readonly<VideoTabProps>) {
  const [videoPreview, setVideoPreview] = useState<VideoPreviewData | null>(null);

  const { uploadVideo, isProcessing } = useVideoProcessing({
    onVideoProcessed: (text: string, filename?: string, videoUrl?: string) => {
      if (filename && videoUrl) {
        setVideoPreview({
          filename,
          extractedText: text,
          videoUrl,
        });
      }
      onVideoProcessed(text, filename, videoUrl);
    },
  });

  const handleClearPreview = () => {
    if (videoPreview?.videoUrl) {
      URL.revokeObjectURL(videoPreview.videoUrl);
    }
    setVideoPreview(null);
  };

  return (
    <div className="space-y-4">
      {videoPreview ? (
        <div className="border rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-medium text-sm">Video Processed</h3>
              <p className="text-xs text-muted-foreground">{videoPreview.filename}</p>
            </div>
            <button
              onClick={handleClearPreview}
              className="text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              Clear
            </button>
          </div>

          <video src={videoPreview.videoUrl} controls className="w-full max-h-40 rounded">
            <track kind="captions" />
          </video>

          <div className="text-xs text-muted-foreground">
            <p className="font-medium mb-1">Extracted Text:</p>
            <p className="line-clamp-3">{videoPreview.extractedText}</p>
          </div>
        </div>
      ) : (
        <>
          <FileDropZone
            onFileSelect={uploadVideo}
            accept="video/*"
            isProcessing={isProcessing}
            icon={Video}
            title="Drop a video here or click to upload"
            description="Speech will be converted to text using AI speech recognition"
            buttonText="Select Video File"
          />
          <ProcessingStatus
            isProcessing={isProcessing}
            message="Processing video... This may take a few minutes"
          />
        </>
      )}
    </div>
  );
}
