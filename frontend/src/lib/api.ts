import { ExtractTextResponse, VideoTextResponse, FactCheckResponse } from '@/types/dashboard/api';

class APIService {
  private readonly baseURL = 'http://127.0.0.1:5000/api';

  private async makeRequest<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const response = await fetch(`${this.baseURL}/${endpoint}`, {
      ...options,
      headers: {
        ...options.headers,
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `Request failed: ${response.statusText}`);
    }

    return response.json();
  }

  async uploadFile<T = unknown>(
    endpoint: string,
    file: File,
    fileFieldName: string = 'file'
  ): Promise<T> {
    const formData = new FormData();
    formData.append(fileFieldName, file);

    return this.makeRequest(endpoint, {
      method: 'POST',
      body: formData,
    });
  }

  async extractImageText(file: File): Promise<ExtractTextResponse> {
    return this.uploadFile('extract-text', file, 'image');
  }

  async extractVideoText(file: File): Promise<VideoTextResponse> {
    return this.uploadFile('extract-video-text', file, 'video');
  }

  async factCheck(
    text: string,
    modelConfig?: {
      provider: string;
      model_id: string;
      temperature?: number;
      max_tokens?: number;
      top_p?: number;
      system_prompt?: string;
    }
  ): Promise<FactCheckResponse> {
    const payload: Record<string, unknown> = { text };

    // Add model configuration if provided
    if (modelConfig) {
      payload.model_config = modelConfig;

      // Frontend logging for verification (dev mode only)
      if (process.env.NODE_ENV === 'development') {
        console.log('🚀 [API] Sending factcheck request with model config:');
        console.log('   Provider:', modelConfig.provider);
        console.log('   Model:', modelConfig.model_id);
        console.log('   Temperature:', modelConfig.temperature);
        console.log('   Max Tokens:', modelConfig.max_tokens);
      }
    }

    return this.makeRequest('factcheck', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });
  }

  async detectAI(text: string): Promise<{ ai_percentage: number }> {
    return this.makeRequest('detect-ai', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });
  }
}

export const apiService = new APIService();
