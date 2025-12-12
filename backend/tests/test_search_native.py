import anyio

from app.core.settings import Settings
from app.features.search.adapters.native import NativeSearchService


def test_native_search_returns_empty_without_keys():
    settings = Settings(
        tavily_api_key="",
        newsapi_api_key="",
        search_provider_paths_csv=(
            "app.features.search.providers.tavily.TavilySearchProvider,"
            "app.features.search.providers.newsapi.NewsApiSearchProvider"
        ),
    )
    search = NativeSearchService(settings=settings, redis=None)

    async def run():
        results = await search.hybrid_search(query="test query", max_results=3)
        assert isinstance(results, list)
        assert results == []

    anyio.run(run)
