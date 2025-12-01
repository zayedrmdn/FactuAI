import { z } from "zod";

// Simple validation schema for basic checks only
const textSchema = z.string()
  .min(10, "Text too short (minimum 10 characters)")
  .max(5000, "Text too long (maximum 5,000 characters)")
  .refine(text => text.trim().length > 0, "Text cannot be empty");

interface ValidationResult {
  isValid: boolean;
  error: string;
  suggestion: string;
}

// Real-time validation - simplified and less restrictive
export function validateBasic(text: string): ValidationResult {
  const cleaned = text.trim();

  // Only show errors if user has typed something
  if (!cleaned) {
    return { isValid: true, error: "", suggestion: "" };
  }

  // Basic length validation
  const basicValidation = textSchema.safeParse(cleaned);
  if (!basicValidation.success) {
    return {
      isValid: false,
      error: basicValidation.error.errors[0].message,
      suggestion: "Please adjust your text length.",
    };
  }

  // Check for obvious gibberish (very relaxed)
  if (isObviousGibberish(cleaned)) {
    return {
      isValid: false,
      error: "Input appears to be random characters or gibberish.",
      suggestion: "Please enter a meaningful statement or question.",
    };
  }

  // Check for excessive repetition (very relaxed)
  if (hasExcessiveRepetition(cleaned)) {
    return {
      isValid: false,
      error: "Input contains too much repetition.",
      suggestion: "Please enter a clear, non-repetitive statement.",
    };
  }

  // Check for obvious spam/promotional content (very relaxed)
  if (isObviousSpam(cleaned)) {
    return {
      isValid: false,
      error: "Content appears to be promotional or spam.",
      suggestion: "Please provide factual content for fact-checking.",
    };
  }

  return { isValid: true, error: "", suggestion: "" };
}

// LLM validation using backend
export async function validateForFactCheck(text: string): Promise<ValidationResult> {
  // First do basic validation
  const basicResult = validateBasic(text);
  if (!basicResult.isValid) {
    return {
      isValid: false,
      error: basicResult.error || "",
      suggestion: basicResult.suggestion || ""
    };
  }

  // Call backend LLM for final validation
  try {
    const response = await fetch("http://localhost:5000/api/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text.trim() }),
    });

    if (!response.ok) {
      throw new Error(`Validation failed: ${response.status}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.warn("LLM validation failed, falling back to basic validation:", error);
    return {
      isValid: basicResult.isValid,
      error: basicResult.error || "",
      suggestion: basicResult.suggestion || ""
    };
  }
}

function isObviousGibberish(text: string): boolean {
  // Only catch very obvious gibberish
  const words = text.split(/\s+/);
  
  // Check for too many single characters
  const singleCharWords = words.filter(word => word.length === 1 && !/[a-zA-Z]/.test(word)).length;
  if (singleCharWords > words.length * 0.3) return true;
  
  // Check for random character sequences (no vowels in long words)
  const longWordsWithoutVowels = words.filter(word => 
    word.length > 4 && !/[aeiouAEIOU]/.test(word)
  ).length;
  if (longWordsWithoutVowels > words.length * 0.2) return true;
  
  return false;
}

function hasExcessiveRepetition(text: string): boolean {
  const words = text.split(/\s+/);
  const wordCount: Record<string, number> = {};

  words.forEach(word => {
    word = word.toLowerCase();
    wordCount[word] = (wordCount[word] || 0) + 1;
  });

  // Only flag if a single word appears more than 50% of the time
  const maxRepetition = Math.max(...Object.values(wordCount));
  return maxRepetition > words.length * 0.5;
}

function isObviousSpam(text: string): boolean {
  const lower = text.toLowerCase();
  
  // Very obvious spam indicators
  const spamPatterns = [
    /\b(click here|buy now|limited time|act now|free money|make money fast|get rich quick)\b/g,
    /\b(www\.|http|\.com|\.org|\.net)\b/g,
    /\b(call now|order now|subscribe now|sign up now)\b/g,
  ];
  
  let spamCount = 0;
  spamPatterns.forEach(pattern => {
    spamCount += (lower.match(pattern) || []).length;
  });
  
  // Only flag if multiple spam indicators
  return spamCount >= 3;
}
function isPromotionalContent(text: string): boolean {
  const lower = text.toLowerCase();
  
  const promoIndicators = [
    /\b(buy|purchase|order|subscribe|sign up|register|download)\b/g,
    /\b(free|discount|sale|offer|deal|limited time|act now|click here)\b/g,
    /\b(visit our|check out our|follow us|like us|share)\b/g,
  ];
  
  let promoCount = 0;
  promoIndicators.forEach(pattern => {
    promoCount += (lower.match(pattern) || []).length;
  });
  
  const words = text.split(/\s+/).length;
  return promoCount > Math.max(2, words * 0.1);
}

function hasFactualContent(text: string): boolean {
  const lower = text.toLowerCase();
  const original = text;

  const factualIndicators = [
    // Numbers and dates
    /\b\d+/g,
    /\b(?:percent|percentage|million|billion|trillion|thousand|hundred|score|dozen|year|years|month|months|week|weeks|day|days|hour|hours|minute|minutes|second|seconds|yesterday|today|tomorrow|january|february|march|april|may|june|july|august|september|october|november|december)\b/g,

    // Reporting language
    /\b(?:according to|study|survey|analysis|research|report|reported|findings|data|evidence|statistics|found|discovered|examined|tested|published|investigation|announced|stated|confirmed|declared|said|says|told|claimed|alleged|revealed|disclosed|admitted|cited)\b/g,

    // Organizations and entities
    /\b(?:company|corporation|firm|organization|institute|institute|university|college|academy|hospital|clinic|court|judiciary|police|military|army|navy|air force|ministry|department|agency|commission|council|association|committee|party|movement|faction)\b/g,

    // News / event verbs
    /\b(?:happened|occurred|took place|released|launched|published|died|killed|born|elected|appointed|arrested|charged|convicted|attacked|bombed|struck|deployed|overthrew|condemned|signed|ratified|voted|passed|failed|launched|created|built|invented|founded|opened|closed)\b/g,

    // Geographic / proper nouns
    /\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b/g,

    // Titles and positions
    /\b(?:president|prime minister|chancellor|governor|mayor|senator|representative|minister|secretary|director|chief|commander|leader|official|spokesperson|ambassador|delegate|commissioner)\b/g,
  ];

  let score = 0;
  factualIndicators.slice(0, -2).forEach(p => score += (lower.match(p) || []).length);
  score += (original.match(factualIndicators[factualIndicators.length-2]) || []).length;
  score += (lower.match(factualIndicators[factualIndicators.length-1]) || []).length;

  const words = text.split(/\s+/).length;
  const threshold = Math.max(1, words * 0.03);
  return score >= threshold;
}

// No changes needed; all validation is already real-time and non-redundant.
