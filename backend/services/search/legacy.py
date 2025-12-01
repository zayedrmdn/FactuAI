# services/search/legacy.py

from .builder import SmartQueryBuilder
from .google_search import GoogleSearchClient
from core.exceptions import SearchError

def extract_key_terms(claim: str) -> str:
    return SmartQueryBuilder().extract_key_terms(claim)

def build_query(claim: str, llm=None) -> str:
    return GoogleSearchClient().build_query(claim, llm)

def google_fetch(question: str, num: int = 2) -> dict:
    return GoogleSearchClient().google_fetch(question, num)
