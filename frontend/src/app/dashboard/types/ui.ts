export interface ValidationResult {
  error?: string;
  suggestion?: string;
  isValid: boolean;
}

export interface FileUploadState {
  isProcessing: boolean;
  error?: string;
  progress?: number;
}

export interface PreviewData {
  filename: string;
  extractedText: string;
  url: string;
}

export interface ComponentProps {
  className?: string;
  children?: React.ReactNode;
}

export type TextSize = 'sm' | 'md' | 'lg';
export type InputType = 'text' | 'image' | 'video';
