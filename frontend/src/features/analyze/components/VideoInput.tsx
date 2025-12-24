import React, { useState } from 'react';
import { Video, FileVideo, X } from 'lucide-react';
import { useVideoProcessing } from '@/lib/hooks/useVideoProcessing';
import { FileDropZone } from '@/components/ui/file-input';
import { ProcessingStatus } from '@/components/ui/feedback-states';
import { Button } from '@/components/ui/button';

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
    <div className="space-y-6">
      {videoPreview ? (
        <div className="bg-card/50 border border-border/50 rounded-xl overflow-hidden animate-in fade-in zoom-in-95 duration-300">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-border/50 bg-muted/20">
            <h3 className="font-semibold text-sm flex items-center gap-2">
              <FileVideo className="h-4 w-4 text-primary" />
              Video Transcribed
            </h3>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClearPreview}
              className="h-7 px-2 text-muted-foreground hover:text-destructive hover:bg-destructive/10 -mr-1"
              title="Clear video"
            >
              <X className="h-3.5 w-3.5 mr-1.5" />
              Clear
            </Button>
          </div>

          <div className="p-4 sm:p-6 grid gap-6 md:grid-cols-2">
            {/* Video Player */}
            <div className="space-y-2">
              <div className="relative rounded-lg overflow-hidden bg-black aspect-video border border-border/50 shadow-sm">
                <video
                  src={videoPreview.videoUrl}
                  controls
                  className="w-full h-full object-contain"
                >
                  <track kind="captions" />
                </video>
              </div>
              <p className="text-xs text-muted-foreground truncate px-1">
                <span className="font-semibold">File:</span> {videoPreview.filename}
              </p>
            </div>

            {/* Transcript */}
            <div className="space-y-2">
              <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Transcript
              </label>
              <div className="relative">
                <div className="p-3 rounded-lg bg-muted/30 border border-border/50 text-sm leading-relaxed max-h-48 sm:max-h-[calc((100vw/16)*9-2rem)] md:max-h-full overflow-y-auto">
                  {videoPreview.extractedText ? (
                    <p className="whitespace-pre-wrap">{videoPreview.extractedText}</p>
                  ) : (
                    <p className="text-muted-foreground italic">No speech detected in video.</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <>
          <FileDropZone
            onFileSelect={uploadVideo}
            accept="video/*"
            isProcessing={isProcessing}
            icon={Video}
            title="Upload Video Analysis"
            description="Upload MP4, MOV, or WEBP. AI will extract speech and analyze visual claims."
            buttonText="Select Video File"
          />
          <ProcessingStatus
            isProcessing={isProcessing}
            message="Transcribing audio and analyzing frames..."
          />
        </>
      )}
    </div>
  );
}
