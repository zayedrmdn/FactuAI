import { useFileUpload } from './useFileUpload';
import { apiService } from '../services/api';
import { VideoTextResponse } from '@/types/dashboard/api';
import { toast } from 'sonner';

interface UseVideoProcessingOptions {
  onVideoProcessed: (text: string, filename?: string, videoUrl?: string) => void;
}

export function useVideoProcessing({ onVideoProcessed }: UseVideoProcessingOptions) {
  const { uploadFile, isProcessing, error, reset } = useFileUpload({
    acceptedTypes: ['video/'],
    maxSize: 100 * 1024 * 1024, // 100MB
    uploadFunction: async (file: File) => {
      toast.info('Processing video... This may take a few minutes');
      return apiService.extractVideoText(file);
    },
    onSuccess: (file: File, result: VideoTextResponse) => {
      if (!result.text || result.text.trim().length === 0) {
        toast.error('No speech found in the video');
        return;
      }

      const videoUrl = URL.createObjectURL(file);
      onVideoProcessed(result.text, file.name, videoUrl);
    },
    onError: (error) => {
      console.error('Video upload error:', error);
    },
    successMessage: 'Text extracted from video speech',
  });

  return {
    uploadVideo: uploadFile,
    isProcessing,
    error,
    reset,
  };
}
