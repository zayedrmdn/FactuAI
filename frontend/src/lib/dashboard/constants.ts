// File size limits
export const MAX_IMAGE_SIZE = 10 * 1024 * 1024; // 10MB
export const MAX_VIDEO_SIZE = 100 * 1024 * 1024; // 100MB
export const MAX_TEXT_LENGTH = 10000;
export const MIN_TEXT_LENGTH = 10;

// Supported file types
export const SUPPORTED_IMAGE_TYPES = ['image/'];
export const SUPPORTED_VIDEO_TYPES = ['video/'];
export const SUPPORTED_TEXT_TYPES = ['.txt', '.pdf'];

// API endpoints
export const API_BASE_URL = 'http://127.0.0.1:8000/api';

// UI constants
export const TEXT_SIZES = {
  sm: 'text-sm',
  md: 'text-base',
  lg: 'text-lg',
} as const;

// Loading messages
export const LOADING_MESSAGES = {
  IMAGE_PROCESSING: 'Extracting text from image...',
  VIDEO_PROCESSING: 'Processing video... This may take a few minutes',
  FACT_CHECKING: 'Fact-checking claims...',
  AI_DETECTION: 'Detecting AI-generated content...',
} as const;
