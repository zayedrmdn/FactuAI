"""Pipeline Summarization Module"""
from pipeline.summarization.evidence_summary import summarise_evidence
from pipeline.summarization.input_summary import summarise_input_text

__all__ = ["summarise_evidence", "summarise_input_text"]
