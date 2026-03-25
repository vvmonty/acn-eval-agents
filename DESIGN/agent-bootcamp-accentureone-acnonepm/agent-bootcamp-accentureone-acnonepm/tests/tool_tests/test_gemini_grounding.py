"""Test web search integration."""

import os

import pytest

from src.utils import pretty_print
from src.utils.tools.gemini_grounding import GeminiGroundingWithGoogleSearch


@pytest.mark.asyncio
async def test_web_search_with_gemini_grounding():
    """Test Gemini grounding with Google Search integration."""
    has_direct = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    has_proxy = bool(os.getenv("WEB_SEARCH_BASE_URL") and os.getenv("WEB_SEARCH_API_KEY"))
    assert has_direct or has_proxy, (
        "Set GOOGLE_API_KEY (or GEMINI_API_KEY) for direct mode, or "
        "WEB_SEARCH_BASE_URL and WEB_SEARCH_API_KEY for proxy mode."
    )

    tool_cls = GeminiGroundingWithGoogleSearch()
    response = await tool_cls.get_web_search_grounded_response(
        "How does the annual growth in the 50th-percentile income "
        "in the US compare with that in Canada?"
    )

    pretty_print(response.text_with_citations)
    assert response.text_with_citations
