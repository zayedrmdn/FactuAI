"""
orchestrator.py
High-level orchestration for building evidence & summaries.

Flow:
1. Collect search + news items          (evidence.candidate.collect_search_items)
2. Fetch articles & extract candidates  (evidence.candidate.extract_candidates)
3. Rank + LLM / semantic selection      (evidence.selector.select_best_evidence)
4. Build source quotes                  (evidence.selector.build_source_quotes)
5. Summaries (evidence + input text)    (summarization.*)

Returned:
    evidence_text, urls, source_quotes
"""

from __future__ import annotations
from typing import List, Dict, Tuple

from pipeline.config import MAX_EVIDENCE_WORDS
from pipeline.evidence.candidate import collect_search_items, extract_candidates
from pipeline.evidence.ranker import rank_sentences
from pipeline.evidence.selector import select_best_evidence, build_source_quotes
from pipeline.summarization import summarise_evidence, summarise_input_text
from core.logging import logger


def build_evidence(
    search_resp: dict,
    claim: str,
    llm=None,
    sents_per_article: int = 5,
    max_google: int = 3,
    max_news: int = 2
) -> Tuple[str, List[str], List[Dict[str, any]]]:
    """
    Orchestrate evidence construction.

    Args:
      search_resp: {"items":[...]} from your search client
      claim: the text claim to verify
      llm: optional LLM client for guided selection
      sents_per_article: how many candidate sentences to pull per source
      max_google: how many Google results to fetch (default 3)
      max_news: how many NewsAPI articles to fetch (default 2)

    Returns:
      evidence_text, list_of_source_urls, list_of_{quote,source,url,score}
    """
    try:
        logger.debug(f"[PIPELINE] build_evidence start claim={claim!r}")

        # 1) collect search + news items
        items = collect_search_items(
            search_resp,
            claim,
            max_google=max_google,
            max_news=max_news
        )

        if not items:
            logger.debug("[PIPELINE] No search items found")
            return "", [], []

        # 2) extract candidate sentences
        candidates, urls = extract_candidates(items, claim, sents_per_article)
        if not candidates:
            logger.debug("[PIPELINE] No candidate sentences extracted")
            return "", urls, []

        # 3) select and concatenate best evidence
        evidence_text = select_best_evidence(
            claim,
            candidates,
            llm=llm,
            max_words=MAX_EVIDENCE_WORDS
        )

        # 4) rank all candidates (so we can attach scores for quoting)
        ranked = rank_sentences(claim, [c["text"] for c in candidates])
        score_map = dict(ranked)
        for c in candidates:
            c["score"] = score_map.get(c["text"], 0.0)

        # 5) build up to three source-quote dicts for the UI
        source_quotes = build_source_quotes(candidates)

        logger.debug(
            f"[PIPELINE] Evidence built: "
            f"words={len(evidence_text.split())}, "
            f"quotes={len(source_quotes)}, urls={len(urls)}"
        )
        return evidence_text, urls, source_quotes

    except Exception as e:
        logger.error(f"[PIPELINE] build_evidence failed: {e}")
        return "", [], []


# -------------------------------------------------------------------------
# Summaries (kept for backward compatibility)
# -------------------------------------------------------------------------

def summarise(evidence_text: str, llm) -> str:
    """Alias for evidence summary."""
    return summarise_evidence(evidence_text, llm)


def summarise_input(text: str, llm) -> str:
    """Alias for input text summary."""
    return summarise_input_text(text, llm)


# -------------------------------------------------------------------------
# Legacy adapters (if any older code imported these names)
# -------------------------------------------------------------------------

def summarise_input_text_legacy(text: str, llm) -> str:
    return summarise_input_text(text, llm)


def semantic_rank(claim: str, sentences: List[str]):
    return rank_sentences(claim, sentences)
