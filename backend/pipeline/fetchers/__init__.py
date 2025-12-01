"""Pipeline Fetchers Module"""
from pipeline.fetchers.newsapi import fetch_newsapi_articles
from pipeline.fetchers.scraping import fetch_article_text, best_sentences

__all__ = [
    "fetch_newsapi_articles",
    "fetch_article_text",
    "best_sentences",
]
