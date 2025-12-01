# services/search/builder.py

import re
import logging
from typing import List, Optional
from dataclasses import dataclass

try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    # Minimal stopword set for fallback
    STOP_WORDS = {
        "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "was",
        "were", "be", "this", "that", "with", "by", "as", "at", "from", "it", "its",
        "his", "her", "their", "has", "have", "had", "about", "into", "over",
        "after", "before", "during"
    }

logger = logging.getLogger(__name__)

@dataclass
class QueryConfig:
    max_terms: int = 5
    min_entity_length: int = 2
    include_numbers: bool = True


class SmartQueryBuilder:
    """
    Extracts up to `max_terms` key terms from a claim to drive
    Google/CSE queries, using spaCy NER when available, else regex.
    """
    def __init__(self, config: Optional[QueryConfig] = None):
        self.config = config or QueryConfig()
        self.nlp = None
        if NLP_AVAILABLE:
            try:
                spacy.prefer_gpu(False)
                self.nlp = spacy.load("en_core_web_sm")
                logger.debug("spaCy model loaded on CPU")
            except OSError:
                logger.info("spaCy model not available; falling back to regex")

    def build_query(self, text: str, llm=None) -> str:
        """Return either a handful of key terms or the full claim as fallback."""
        logger.debug(f"Building query for: {text!r}")
        terms = self._extract_key_terms(text)
        if len(terms.split()) >= 2:
            logger.debug(f"→ using terms: {terms}")
            return terms
        logger.debug("→ fallback to full claim")
        return text

    def _extract_key_terms(self, text: str) -> str:
        if self.nlp:
            return self._from_spacy(text)
        return self._from_regex(text)

    def _from_spacy(self, text: str) -> str:
        doc = self.nlp(text)
        terms: List[str] = []

        # 1) Named entities of interest
        for ent in doc.ents:
            if ent.label_ in {
                "PERSON", "ORG", "PRODUCT", "EVENT", "GPE",
                "DATE", "PERCENT", "MONEY", "CARDINAL"
            }:
                terms.append(ent.text)

        # 2) Nouns and proper nouns (non‑stop, min length)
        for tok in doc:
            if tok.pos_ in ("NOUN", "PROPN") \
               and not tok.is_stop \
               and len(tok.text) >= self.config.min_entity_length:
                terms.append(tok.text)

        # 3) Any domain‑specific patterns (model names, verbs, etc.)
        terms.extend(self._domain_patterns(text))
        return self._dedupe_and_limit(terms)

    def _from_regex(self, text: str) -> str:
        terms: List[str] = []
        terms.extend(self._domain_patterns(text))

        # Proper‑noun phrases (Up to 3)
        proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        terms.extend(proper_nouns[:3])

        # Optional number‑based terms
        if self.config.include_numbers:
            terms.extend(re.findall(r"\b\d{4}\b", text))          # years
            terms.extend(re.findall(r"\b\d+(?:\.\d+)?%?\b", text)[:2])

        # If still too few, grab any other 3+ letter words
        if len([t for t in terms if re.search(r"[A-Za-z]", t)]) < 2:
            words = re.findall(r"\b[A-Za-z]{3,}\b", text)
            extra = [w for w in words if w.lower() not in STOP_WORDS][:3]
            terms.extend(extra)

        return self._dedupe_and_limit(terms)

    def _domain_patterns(self, text: str) -> List[str]:
        """Look for known model names, OpenAI, and action verbs."""
        terms: List[str] = []
        low = text.lower()

        if "openai" in low:
            terms.append("OpenAI")

        # AI model patterns
        patterns = [
            (r"\b(gpt[- ]?\d+(?:\.\d+)?)\b", lambda m: m.group(1).upper()),
            (r"\b(claude[- ]?\d*)\b",     lambda m: m.group(1).title()),
            (r"\b(gemini[- ]?\d*)\b",     lambda m: m.group(1).title()),
            (r"\b(llama[- ]?\d*)\b",      lambda m: m.group(1).upper()),
        ]
        for pat, fmt in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                terms.append(fmt(m))

        # Common action verbs for launching/releasing
        verbs = re.findall(
            r"\b(released|launched|announced|acquired|merged|resigned|arrested)\b",
            low
        )
        terms.extend(verbs[:2])

        return terms

    def _dedupe_and_limit(self, terms: List[str]) -> str:
        """Deduplicate (case‑insensitive) and cap at max_terms."""
        seen = set()
        out: List[str] = []
        for t in terms:
            tc = t.strip()
            key = tc.lower()
            if tc and key not in seen:
                out.append(tc)
                seen.add(key)
                if len(out) >= self.config.max_terms:
                    break
        return " ".join(out)
