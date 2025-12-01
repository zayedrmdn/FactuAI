import { ExtractTextResponse, VideoTextResponse, FactCheckResponse } from '../types/api';

class APIService {
  private baseURL = 'http://localhost:5000/api';
  
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
  
  async uploadFile(endpoint: string, file: File, fileFieldName: string = 'file'): Promise<any> {
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
  
  async factCheck(text: string): Promise<FactCheckResponse> {
    return this.makeRequest('factcheck', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
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
