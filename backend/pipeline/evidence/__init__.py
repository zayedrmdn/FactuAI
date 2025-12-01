"""Pipeline Evidence Module"""
from pipeline.evidence.candidate import collect_search_items, extract_candidates
from pipeline.evidence.ranker import rank_sentences
from pipeline.evidence.selector import select_best_evidence, build_source_quotes

__all__ = [
    "collect_search_items",
    "extract_candidates", 
    "rank_sentences",
    "select_best_evidence",
    "build_source_quotes",
]
