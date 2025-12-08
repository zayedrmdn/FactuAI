export interface APIResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface ExtractTextResponse {
  text: string;
  ai_percentage?: number;
  ai_error?: string;
}

export interface VideoTextResponse {
  text: string;
}

export interface FactCheckResponse {
  results: FactCheckResult[];
  summary: string;
  updated: string;
  average_confidence: number;
}

export interface FactCheckResult {
  claim: string;
  verdict: string;
  confidence: number;
  evidence: string[];
  sources: string[];
}
