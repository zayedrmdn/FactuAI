export interface SourceQuote {
  quote: string;
  source: string;
  url: string;
}

export interface FactCheckResult {
  claim: string;
  label: string;
  confidence: number;
  evidence: string;
  sources: string[];
  explanation?: string;
  reasoning?: string;
  source_quotes?: SourceQuote[];
}

export interface QAResult {
  type: 'person_info' | 'general_qa';
  question: string;
  answer: string;
  sources: string[];
  /** 0–1 confidence score from backend */
  confidence: number;
}

export interface FactCheckSummary {
  summary: string;
  updated: string;
  average_confidence: number;
}

export interface LoadingPhase {
  phase: 'summary' | 'factcheck' | null;
  progress?: number;
  currentClaim?: string;
}

export interface HistoryItem {
  id: string;
  input: string;
  results: FactCheckResult[];
  summary: string;
  timestamp: string;
  type: 'text' | 'image' | 'video';
  metadata?: {
    filename?: string;
    imageUrl?: string;
    videoUrl?: string | undefined;
    aiScore?: number | undefined;
  };
}
