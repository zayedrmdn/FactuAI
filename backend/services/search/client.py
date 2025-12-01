# services/search/client.py

import requests
from typing import Any, Dict, List, Optional

# detect if the official Google API client is available
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False

from .builder import SmartQueryBuilder, QueryConfig
from .base import SearchInterface
from core.exceptions import SearchError
from core.logging import logger
from pipeline.config import GOOGLE_API_KEY, GOOGLE_CX_ID


class GoogleSearchClient(SearchInterface):
    def __init__(
        self,
        api_key: str = "",
        cx_id: str = "",
        config: Optional[QueryConfig] = None
    ):
        self.api_key = api_key or GOOGLE_API_KEY
        self.cx_id = cx_id or GOOGLE_CX_ID
        self.query_builder = SmartQueryBuilder(config)
        self._service = None

        if self.is_available() and GOOGLE_API_AVAILABLE:
            try:
                self._service = build("customsearch", "v1", developerKey=self.api_key)
                logger.debug("Initialized official Google Custom Search client")
            except Exception as e:
                logger.error(f"Failed to initialize Google API client: {e}")

    def is_available(self) -> bool:
        return bool(self.api_key and self.cx_id)

    def get_search_info(self) -> Dict[str, Any]:
        return {
            "service": "Google Custom Search",
            "api_key_configured": bool(self.api_key),
            "cx_id_configured": bool(self.cx_id),
            "official_client": bool(self._service),
            "nlp_available": bool(self.query_builder.nlp),
        }

    def build_query(self, claim: str, llm=None) -> str:
        return self.query_builder.build_query(claim, llm)

    def search(self, query: str, num_results: int = 3) -> List[str]:
        if not self.is_available():
            raise SearchError("Google API credentials not configured")

        try:
            if self._service:
                resp = self._service.cse().list(
                    q=query, cx=self.cx_id, num=num_results
                ).execute()
                items = resp.get("items", [])
            else:
                resp = requests.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params={
                        "key": self.api_key,
                        "cx": self.cx_id,
                        "q": query,
                        "num": num_results
                    },
                    timeout=10
                ).json()
                if resp.get("error"):
                    raise SearchError(resp["error"].get("message", "Unknown API error"))
                items = resp.get("items", [])

            # Return just the URLs
            urls = [
                it.get("link", "").strip()
                for it in items
                if it.get("link", "").startswith(("http://", "https://"))
            ]
            logger.debug(f"[GOOGLE] Retrieved {len(urls)} URLs")
            return urls

        except Exception as e:
            logger.error(f"Google search failed: {e}")
            raise SearchError(str(e))


    def _search_official(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        logger.debug(f"[OFFICIAL] Searching Google for: '{query}' (num={num_results})")
        try:
            resp = self._service.cse().list(q=query, cx=self.cx_id, num=num_results).execute()
            items = resp.get("items", [])
        except HttpError as e:
            raise SearchError(f"Google API error: {e.resp.status}")
        return self._filter_items(items)

    def _search_requests(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        logger.debug(f"[REQUESTS] Searching Google via REST for: '{query}' (num={num_results})")
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": self.api_key, "cx": self.cx_id, "q": query, "num": num_results},
            timeout=10
        )
        data = resp.json()
        if data.get("error"):
            raise SearchError(data["error"].get("message", "Unknown API error"))
        return self._filter_items(data.get("items", []))

    def _filter_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid = []
        for it in items or []:
            url = it.get("link", "").strip()
            if url.startswith(("http://", "https://")):
                valid.append({
                    "title":   it.get("title", "").strip(),
                    "url":     url,
                    "snippet": it.get("snippet", "").strip()
                })
        logger.debug(f"Filtered {len(valid)} valid results out of {len(items or [])}")
        return valid

    def google_fetch(self, question: str, num: int = 2) -> Dict[str, Any]:
        try:
            results = self.search(question, num)
            return {"items": results}
        except SearchError as e:
            return {"error": {"message": str(e)}}