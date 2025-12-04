"""
selector.py
Evidence selection logic using semantic ranking and optional LLM guidance.
"""

import re
from typing import List, Dict, Any, Optional

from pipeline.config import MAX_EVIDENCE_WORDS, SIMILARITY_THRESHOLD
from pipeline.evidence.ranker import rank_sentences
from core.logging import logger


def _safe_log_string(text: str) -> str:
    """
    Safely encode a string for logging on Windows consoles.
    Replaces non-ASCII characters with '?' to prevent UnicodeEncodeError.
    """
    return text.encode('ascii', 'replace').decode('ascii')


def select_best_evidence(
    claim: str,
    candidates: List[Dict[str, Any]],
    llm=None,
    max_words: int = MAX_EVIDENCE_WORDS
) -> str:
    """
    1) Rank all candidates by semantic similarity.
    2) Keep top 5 by score.
    3) Ask LLM to pick 1-2 by index.
    4) Fallback: join top 3 until word budget.
    """
    if not candidates:
        return ""

    # 1) Semantic ranking
    texts = [c["text"] for c in candidates]
    ranked = rank_sentences(claim, texts)
    score_map = {txt: score for txt, score in ranked}
    for c in candidates:
        c["score"] = score_map.get(c["text"], 0.0)

    # 2) Keep the top-5 sentences
    top5 = sorted(candidates, key=lambda c: c["score"], reverse=True)[:5]

    # 3) Try LLM-driven pick
    if llm and len(top5) >= 2:
        try:
            logger.info(f"[SELECTOR] Asking LLM to pick from top {len(top5)}")
            prompt = (
                f"Claim to evaluate: \"{claim}\"\n\n"
                "Carefully analyze the claim and the provided candidate sentences (labeled 1-5). "
                "Your goal is to identify 1 or 2 sentences that offer the most direct and conclusive evidence, either supporting or refuting the claim. "
                "Consider the following criteria:\n"
                "- Directness: Does the sentence directly address the core assertion of the claim?\n"
                "- Conclusiveness: Does the sentence provide a clear and unambiguous answer (support or refute), or does it merely offer related information?\n"
                "- Relevance: Is the sentence highly relevant to the specific details of the claim?\n\n"
                "If multiple sentences meet these criteria, choose the 1 or 2 that are strongest. "
                "If no sentence directly verifies or refutes the claim, or if they are tangential, reply 'NONE'.\n\n"
                "Candidate Sentences:\n"
                + "\n".join(f"{i+1}. {c['text']}" for i, c in enumerate(top5)) + "\n\n"
                "Based on your analysis, provide ONLY the numbers of the selected sentences, separated by commas (e.g., '1, 3'). If NONE, just reply 'NONE'."
            )
            logger.debug(f"[SELECTOR] LLM prompt: {_safe_log_string(prompt)}")

            resp = llm.generate_response(prompt, max_tokens=200).strip().upper()
            if not resp.startswith("NONE"):
                picks = []
                for idx in map(int, re.findall(r"\d+", resp)):
                    if 1 <= idx <= len(top5):
                        picks.append(top5[idx-1]["text"])

                if picks:
                    # dedupe and respect max_words
                    uniq = []
                    for p in picks:
                        if p not in uniq:
                            uniq.append(p)
                    joined = " ".join(uniq).split()
                    return " ".join(joined[:max_words])
        except Exception as e:
            logger.error(f"[SELECTOR] LLM selection error: {e}")

    # 4) Fallback: take top-3 until max_words
    selected = []
    wc = 0
    for c in top5[:3]:
        wcount = len(c["text"].split())
        if wc + wcount <= max_words:
            selected.append(c["text"])
            wc += wcount
        else:
            break

    return " ".join(selected)


def build_source_quotes(
    candidates: List[Dict[str, Any]],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    From scored candidates, return up to top_k quotes with metadata.
    
    Returns list of dicts with: quote, source, url, score
    """
    if not candidates:
        return []

    # Sort by score descending
    sorted_candidates = sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)
    
    quotes = []
    seen_urls = set()
    
    for c in sorted_candidates:
        url = c.get("url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        
        quotes.append({
            "quote": c.get("text", ""),
            "source": c.get("title", c.get("source", "Unknown")),
            "url": url,
            "score": c.get("score", 0.0)
        })
        
        if len(quotes) >= top_k:
            break
    
    return quotes
