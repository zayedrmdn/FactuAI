import { useState, useCallback } from 'react';
import { toast } from 'sonner';
import { FileUploadState } from '../types/ui';

interface UseFileUploadOptions<T> {
  acceptedTypes: string[];
  maxSize?: number;
  onSuccess: (file: File, result: T) => void;
  onError?: (error: string) => void;
  uploadFunction: (file: File) => Promise<T>;
  successMessage?: string;
}

export function useFileUpload<T>(options: UseFileUploadOptions<T>) {
  const [state, setState] = useState<FileUploadState>({
    isProcessing: false,
  });
  
  const uploadFile = useCallback(async (file: File) => {
    if (state.isProcessing) {
      console.log("Already processing, skipping duplicate call");
      return;
    }
    
    // Validate file type
    if (!options.acceptedTypes.some(type => file.type.startsWith(type))) {
      const error = `Please select a valid file type`;
      toast.error(error);
      options.onError?.(error);
      return;
    }
    
    // Validate file size
    if (options.maxSize && file.size > options.maxSize) {
      const error = `File size exceeds ${options.maxSize / 1024 / 1024}MB limit`;
      toast.error(error);
      options.onError?.(error);
      return;
    }
    
    setState({ isProcessing: true });
    
    try {
      const result = await options.uploadFunction(file);
      options.onSuccess(file, result);
      
      if (options.successMessage) {
        toast.success(options.successMessage);
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Upload failed';
      setState({ isProcessing: false, error: errorMsg });
      toast.error(errorMsg);
      options.onError?.(errorMsg);
    } finally {
      setState(prev => ({ ...prev, isProcessing: false }));
    }
  }, [state.isProcessing, options]);
  
  const reset = useCallback(() => {
    setState({ isProcessing: false });
  }, []);
  
  return { 
    uploadFile, 
    isProcessing: state.isProcessing,
    error: state.error,
    reset 
  };
}
