export interface SourceQuote {
  quote: string;
  source: string;
  url: string;
}

/**
 * Evidence item as returned by the backend API.
 * Maps to backend/app/features/analyze/schemas.py:EvidenceItem
 */
export interface EvidenceItem {
  snippet: string;
  source_url: string;
  source_title?: string;
  source_domain: string;
  relevance_score: number;
}

/**
 * Raw claim result from backend API.
 * Maps to backend/app/features/analyze/schemas.py:ClaimResult
 */
export interface FactCheckApiResult {
  claim_text: string;
  verdict: 'true' | 'false' | 'mostly_true' | 'mostly_false' | 'mixed' | 'unverifiable';
  confidence: number;
  reasoning: string;
  evidence: EvidenceItem[];
}

/**
 * Frontend-friendly result format used by UI components.
 * Mapped from FactCheckApiResult via mapApiResultToFactCheckResult()
 */
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

/**
 * Transform backend API result to frontend-friendly format.
 * This bridges the data contract between backend and frontend.
 */
export function mapApiResultToFactCheckResult(apiResult: FactCheckApiResult): FactCheckResult {
  // Extract source URLs from evidence items
  const sources = apiResult.evidence.map((e) => e.source_url);

  // Create evidence text from snippets
  const evidenceText = apiResult.evidence.map((e) => e.snippet).join(' ');

  // Create source quotes from evidence items
  const sourceQuotes: SourceQuote[] = apiResult.evidence.map((e) => ({
    quote: e.snippet,
    source: e.source_title || e.source_domain,
    url: e.source_url,
  }));

  return {
    claim: apiResult.claim_text,
    label: apiResult.verdict,
    confidence: apiResult.confidence,
    evidence: evidenceText,
    sources,
    reasoning: apiResult.reasoning,
    source_quotes: sourceQuotes,
  };
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
