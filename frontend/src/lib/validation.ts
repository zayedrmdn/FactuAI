import { ValidationResult } from '@/types/dashboard/ui';

export function validateBasic(input: string): ValidationResult {
  const trimmed = input.trim();

  // Only show errors if user has typed something
  if (!trimmed) {
    return { isValid: true };
  }

  // Basic length validation
  if (trimmed.length < 10) {
    return {
      error: 'Text too short',
      suggestion: 'Please enter at least 10 characters',
      isValid: false,
    };
  }

  if (trimmed.length > 5000) {
    return {
      error: 'Text too long',
      suggestion: 'Please keep text under 5,000 characters',
      isValid: false,
    };
  }

  // Check for obvious gibberish (very relaxed)
  if (isObviousGibberish(trimmed)) {
    return {
      error: 'Input appears to be random characters or gibberish',
      suggestion: 'Please enter a meaningful statement or question',
      isValid: false,
    };
  }

  // Check for excessive repetition (very relaxed)
  if (hasExcessiveRepetition(trimmed)) {
    return {
      error: 'Input contains too much repetition',
      suggestion: 'Please enter a clear, non-repetitive statement',
      isValid: false,
    };
  }

  // Check for obvious spam/promotional content (very relaxed)
  if (isObviousSpam(trimmed)) {
    return {
      error: 'Content appears to be promotional or spam',
      suggestion: 'Please provide factual content for fact-checking',
      isValid: false,
    };
  }

  return { isValid: true };
}

function isObviousGibberish(text: string): boolean {
  // Only catch very obvious gibberish
  const words = text.split(/\s+/);

  // Check for too many single characters
  const singleCharWords = words.filter(
    (word) => word.length === 1 && !/[a-zA-Z]/.test(word)
  ).length;
  if (singleCharWords > words.length * 0.3) return true;

  // Check for random character sequences (no vowels in long words)
  const longWordsWithoutVowels = words.filter(
    (word) => word.length > 4 && !/[aeiouAEIOU]/.test(word)
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

// Note: isPromotionalContent and hasFactualContent are not exported
// and appear to be unused. Keeping minimal implementations for reference.

function isPromotionalContent(text: string): boolean {
  const lower = text.toLowerCase();

  // Split into smaller patterns to reduce complexity
  const promoIndicators = [
    /\b(buy|purchase|order|subscribe)\b/g,
    /\b(sign up|register|download)\b/g,
    /\b(free|discount|sale|offer|deal)\b/g,
    /\b(limited time|act now|click here)\b/g,
    /\b(visit our|check out our|follow us|like us|share)\b/g,
  ];

  let promoCount = 0;
  for (const pattern of promoIndicators) {
    promoCount += countMatches(lower, pattern);
  }

  const words = text.split(/\s+/).length;
  return promoCount > Math.max(2, words * 0.1);
}

function hasFactualContent(text: string): boolean {
  const lower = text.toLowerCase();
  const original = text;

  // Split complex patterns into smaller ones (max 20 alternatives each)
  const numberPattern = /\b\d+/g;

  // Time/quantity words - split into groups
  const timeWords1 = /\b(percent|percentage|million|billion|trillion|thousand|hundred)\b/g;
  const timeWords2 = /\b(year|years|month|months|week|weeks|day|days)\b/g;
  const timeWords3 = /\b(hour|hours|minute|minutes|second|seconds)\b/g;
  const timeWords4 = /\b(yesterday|today|tomorrow)\b/g;
  const monthWords =
    /\b(january|february|march|april|may|june|july|august|september|october|november|december)\b/g;

  // Attribution words - split into groups
  const attribution1 = /\b(according to|study|survey|analysis|research|report|reported)\b/g;
  const attribution2 = /\b(findings|data|evidence|statistics|found|discovered)\b/g;
  const attribution3 = /\b(examined|tested|published|investigation|announced|stated)\b/g;
  const attribution4 = /\b(confirmed|declared|said|says|told|claimed|alleged)\b/g;
  const attribution5 = /\b(revealed|disclosed|admitted|cited)\b/g;

  // Organizations - split into groups
  const org1 = /\b(company|corporation|firm|organization|institute|university)\b/g;
  const org2 = /\b(college|academy|hospital|clinic|court|judiciary)\b/g;
  const org3 = /\b(police|military|army|navy|air force|ministry)\b/g;
  const org4 = /\b(department|agency|commission|council|association|committee)\b/g;

  // Events - split into groups
  const events1 = /\b(happened|occurred|took place|released|launched|published)\b/g;
  const events2 = /\b(died|killed|born|elected|appointed|arrested)\b/g;
  const events3 = /\b(charged|convicted|attacked|bombed|struck|deployed)\b/g;
  const events4 = /\b(overthrew|condemned|signed|ratified|voted|passed)\b/g;
  const events5 = /\b(failed|created|built|invented|founded|opened|closed)\b/g;

  // Proper nouns (capitalized words)
  const properNouns = /\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b/g;

  // Titles - split into groups
  const titles1 = /\b(president|prime minister|chancellor|governor|mayor)\b/g;
  const titles2 = /\b(senator|representative|minister|secretary|director)\b/g;
  const titles3 = /\b(chief|commander|leader|official|spokesperson)\b/g;
  const titles4 = /\b(ambassador|delegate|commissioner)\b/g;

  let score = 0;

  // Count matches from all patterns
  const lowerPatterns = [
    numberPattern,
    timeWords1,
    timeWords2,
    timeWords3,
    timeWords4,
    monthWords,
    attribution1,
    attribution2,
    attribution3,
    attribution4,
    attribution5,
    org1,
    org2,
    org3,
    org4,
    events1,
    events2,
    events3,
    events4,
    events5,
    titles1,
    titles2,
    titles3,
    titles4,
  ];

  for (const pattern of lowerPatterns) {
    score += countMatches(lower, pattern);
  }

  // Add proper nouns from original text
  score += countMatches(original, properNouns);

  const words = text.split(/\s+/).length;
  const threshold = Math.max(1, words * 0.03);
  return score >= threshold;
}

// Export for potential future use
export { isPromotionalContent, hasFactualContent };
