"""Search client abstraction for ResearcherAgent."""

import logging

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client.

    Uses Tavily when `TAVILY_API_KEY` is configured. Without a key it falls back to a
    deterministic offline mock so Researcher/Analyst/Writer stay testable without network
    access or paid credentials.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.tavily_api_key

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""

        if not self._api_key:
            logger.warning("TAVILY_API_KEY not set; returning offline mock search results")
            return self._search_mock(query, max_results)
        return self._search_tavily(query, max_results)

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        from tavily import TavilyClient  # type: ignore[import-not-found]

        client = TavilyClient(api_key=self._api_key)
        response = client.search(query=query, max_results=max_results)
        results = response.get("results", [])
        return [
            SourceDocument(
                title=item.get("title") or "Untitled",
                url=item.get("url"),
                snippet=(item.get("content") or "")[:500],
                metadata={"score": item.get("score")},
            )
            for item in results
        ]

    def _search_mock(self, query: str, max_results: int) -> list[SourceDocument]:
        return [
            SourceDocument(
                title=f"Mock source {i + 1} for '{query}'",
                url=f"https://example.com/mock-source-{i + 1}",
                snippet=(
                    f"Offline mock snippet {i + 1} discussing '{query}'. Set TAVILY_API_KEY "
                    "to use real search results instead."
                ),
                metadata={"mock": True, "rank": i + 1},
            )
            for i in range(max_results)
        ]
