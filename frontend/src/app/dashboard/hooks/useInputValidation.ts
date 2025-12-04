"use client";

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

/** Count regex matches using exec() */
function countMatches(text: string, pattern: RegExp): number {
  let count = 0;
  // Reset lastIndex for global regex
  pattern.lastIndex = 0;
  while (pattern.exec(text) !== null) {
    count++;
  }
  return count;
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
    const response = await fetch("/api/validate", {
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

  for (const w of words) {
    const word = w.toLowerCase();
    wordCount[word] = (wordCount[word] || 0) + 1;
  }

  // Only flag if a single word appears more than 50% of the time
  const maxRepetition = Math.max(...Object.values(wordCount));
  return maxRepetition > words.length * 0.5;
}

function isObviousSpam(text: string): boolean {
  const lower = text.toLowerCase();
  
  // Very obvious spam indicators - split into smaller patterns
  const spamPatterns = [
    /\b(click here|buy now|limited time|act now)\b/g,
    /\b(free money|make money fast|get rich quick)\b/g,
    /\b(www\.|http|\.com|\.org|\.net)\b/g,
    /\b(call now|order now|subscribe now|sign up now)\b/g,
  ];
  
  let spamCount = 0;
  for (const pattern of spamPatterns) {
    spamCount += countMatches(lower, pattern);
  }
  
  // Only flag if multiple spam indicators
  return spamCount >= 3;
}
