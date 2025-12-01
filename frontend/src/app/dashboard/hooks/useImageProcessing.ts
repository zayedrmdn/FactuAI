import { useFileUpload } from './useFileUpload';
import { apiService } from '../services/api';
import { ExtractTextResponse } from '../types/api';

interface UseImageProcessingOptions {
  onImageProcessed: (text: string, aiScore: number | null, imageUrl: string, aiError?: string) => void;
}

export function useImageProcessing({ onImageProcessed }: UseImageProcessingOptions) {
  const { uploadFile, isProcessing, error, reset } = useFileUpload({
    acceptedTypes: ['image/'],
    maxSize: 10 * 1024 * 1024, // 10MB
    uploadFunction: async (file: File) => {
      return apiService.extractImageText(file);
    },
    onSuccess: (file: File, result: ExtractTextResponse) => {
      const imageUrl = URL.createObjectURL(file);
      onImageProcessed(
        result.text || '',
        result.ai_percentage || null,
        imageUrl,
        result.ai_error
      );
    },
    successMessage: "Text extracted from image"
  });

  return {
    uploadImage: uploadFile,
    isProcessing,
    error,
    reset
  };
}
