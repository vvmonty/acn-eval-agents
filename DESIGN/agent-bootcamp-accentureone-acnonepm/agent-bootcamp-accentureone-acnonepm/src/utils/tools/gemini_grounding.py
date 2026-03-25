"""Implements a tool to fetch Google Search grounded responses from Gemini."""

import asyncio
import os
from typing import Any, Literal, cast
from urllib.parse import urlparse

import backoff
import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel
from pydantic.fields import Field


RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _gemini_developer_api_key() -> str | None:
    """API key for the Gemini developer API (Google AI Studio)."""
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


class ModelSettings(BaseModel):
    """Configuration for the Gemini model used for web search."""

    model: Literal["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"] = (
        "gemini-2.5-flash"
    )
    temperature: float | None = Field(default=0.2, ge=0, le=2)
    max_output_tokens: int | None = Field(default=None, ge=1)
    seed: int | None = None
    thinking_budget: int | None = Field(default=-1, ge=-1)


class GroundedResponse(BaseModel):
    """Response returned by Gemini."""

    text_with_citations: str
    web_search_queries: list[str]
    citations: dict[int, str]


class GeminiGroundingWithGoogleSearch:
    """Google Search–grounded answers via Gemini.

    - **Direct**: ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY`` — native ``google.genai`` with
      the Google Search tool (recommended for bootcamp / AI Studio).
    - **Proxy**: ``WEB_SEARCH_BASE_URL`` + ``WEB_SEARCH_API_KEY`` — HTTP proxy to a
      grounding endpoint (legacy).
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        model_settings: ModelSettings | None = None,
        max_concurrency: int = 5,
        timeout: int = 300,
    ) -> None:
        self.model_settings = model_settings or ModelSettings()
        self._semaphore = asyncio.Semaphore(max_concurrency)

        dev_key = _gemini_developer_api_key()
        self.base_url = base_url or os.getenv("WEB_SEARCH_BASE_URL")
        self.api_key = api_key or os.getenv("WEB_SEARCH_API_KEY")

        if dev_key:
            self._mode: Literal["direct", "proxy"] = "direct"
            self._genai_client = genai.Client(api_key=dev_key)
            self._client = None
            self._endpoint = ""
            return

        if self.api_key is None:
            raise ValueError(
                "Set GOOGLE_API_KEY or GEMINI_API_KEY for direct Gemini search "
                "grounding, or set WEB_SEARCH_API_KEY for proxy mode."
            )
        if self.base_url is None:
            raise ValueError(
                "WEB_SEARCH_BASE_URL is not set (required for proxy mode when no "
                "GOOGLE_API_KEY / GEMINI_API_KEY is set)."
            )

        self._mode = "proxy"
        self._genai_client = None
        self._client = httpx.AsyncClient(
            timeout=timeout, headers={"X-API-Key": self.api_key}
        )
        self._endpoint = f"{self.base_url.strip('/')}/api/v1/grounding_with_search"

    def _direct_generate_sync(self, query: str) -> GroundedResponse:
        """Gemini + Google Search (sync; use from thread pool in async callers)."""
        assert self._genai_client is not None
        ms = self.model_settings
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=ms.temperature,
            max_output_tokens=ms.max_output_tokens,
            seed=ms.seed,
        )
        response = self._genai_client.models.generate_content(
            model=ms.model,
            contents=query,
            config=config,
        )
        response_json = cast(
            dict[str, Any],
            response.model_dump(mode="json", exclude_none=True),
        )
        candidates = response_json.get("candidates")
        first = candidates[0] if candidates else None
        grounding_metadata: dict[str, Any] | None = (
            first.get("grounding_metadata") if isinstance(first, dict) else None
        )
        web_search_queries: list[str] = []
        if isinstance(grounding_metadata, dict):
            raw_q = grounding_metadata.get("web_search_queries")
            if isinstance(raw_q, list):
                web_search_queries = [str(x) for x in raw_q]

        text_with_citations, citations = add_citations(response_json)
        if not text_with_citations.strip():
            text_with_citations = (getattr(response, "text", None) or "").strip()

        return GroundedResponse(
            text_with_citations=text_with_citations,
            web_search_queries=web_search_queries,
            citations=citations,
        )

    def grounded_search_text_sync(self, query: str) -> str:
        """Return grounded text for a synchronous ``agents.function_tool`` (direct API only)."""
        if self._mode != "direct":
            raise RuntimeError(
                "grounded_search_text_sync needs GOOGLE_API_KEY / GEMINI_API_KEY. "
                "For proxy mode use get_web_search_grounded_response."
            )
        return self._direct_generate_sync(query).text_with_citations or ""

    async def get_web_search_grounded_response(self, query: str) -> GroundedResponse:
        """Google Search grounded response (async)."""
        if self._mode == "direct":
            async with self._semaphore:
                return await asyncio.to_thread(self._direct_generate_sync, query)

        payload = self.model_settings.model_dump(exclude_unset=True)
        payload["query"] = query

        response = await self._post_payload(payload)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise exc from exc

        response_json = response.json()

        candidates: list[dict[str, Any]] | None = response_json.get("candidates")
        grounding_metadata: dict[str, Any] | None = (
            candidates[0].get("grounding_metadata") if candidates else None
        )
        web_search_queries: list[str] = []
        if isinstance(grounding_metadata, dict):
            raw_q = grounding_metadata.get("web_search_queries")
            if isinstance(raw_q, list):
                web_search_queries = [str(x) for x in raw_q]

        text_with_citations, citations = add_citations(response_json)

        return GroundedResponse(
            text_with_citations=text_with_citations,
            web_search_queries=web_search_queries,
            citations=citations,
        )

    @backoff.on_exception(
        backoff.expo,
        (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.HTTPStatusError,
        ),
        giveup=lambda exc: (
            isinstance(exc, httpx.HTTPStatusError)
            and exc.response.status_code not in RETRYABLE_STATUS
        ),
        jitter=backoff.full_jitter,
        max_tries=5,
    )
    async def _post_payload(self, payload: dict[str, object]) -> httpx.Response:
        async with self._semaphore:
            return await self._client.post(self._endpoint, json=payload)


_DEFAULT_GROUNDING: GeminiGroundingWithGoogleSearch | None = None


def _model_settings_from_gemini_grounding_env() -> ModelSettings:
    raw = os.getenv("GEMINI_GROUNDING_MODEL", "").strip()
    if raw in ("gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"):
        return ModelSettings(
            model=cast(
                Literal["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
                raw,
            )
        )
    return ModelSettings()


def _default_grounding_singleton() -> GeminiGroundingWithGoogleSearch:
    global _DEFAULT_GROUNDING
    if _DEFAULT_GROUNDING is None:
        _DEFAULT_GROUNDING = GeminiGroundingWithGoogleSearch(
            model_settings=_model_settings_from_gemini_grounding_env(),
        )
    return _DEFAULT_GROUNDING


def google_search_grounded_sync(query: str) -> str:
    """Web search tool body: uses :class:`GeminiGroundingWithGoogleSearch` (direct API).

    Register with ``agents.function_tool(..., name_override="search_web")`` beside
    OpenAI-compat chat. Same API key as Gemini; grounding runs via native ``genai``.
    """
    return _default_grounding_singleton().grounded_search_text_sync(query)


def add_citations(response: dict[str, object]) -> tuple[str, dict[int, str]]:
    """Add inline citation links to grounded text."""
    candidates = response.get("candidates") if isinstance(response, dict) else None
    if not candidates:
        return "", {}

    candidate = candidates[0] or {}
    content = candidate.get("content") if isinstance(candidate, dict) else {}
    parts = content.get("parts") if isinstance(content, dict) else []

    text_parts: list[str] = []
    for part in parts if isinstance(parts, list) else []:
        if isinstance(part, dict):
            t = part.get("text")
            if isinstance(t, str) and t.strip():
                text_parts.append(t.strip())
    text = "\n\n".join(text_parts)
    if not text:
        return "", {}

    meta = candidate.get("grounding_metadata") if isinstance(candidate, dict) else {}
    raw_supports = meta.get("grounding_supports") if isinstance(meta, dict) else []
    supports = raw_supports if isinstance(raw_supports, list) else []
    raw_chunks = meta.get("grounding_chunks") if isinstance(meta, dict) else []
    chunks = raw_chunks if isinstance(raw_chunks, list) else []

    citations: dict[int, str] = {}
    chunk_to_id: dict[int, int] = {}

    if supports and chunks:
        citations, chunk_to_id = _collect_citations(candidate)

    sorted_supports = sorted(
        (s for s in supports if isinstance(s, dict) and s.get("segment")),
        key=lambda s: s["segment"].get("end_index", 0),
        reverse=True,
    )

    for support in sorted_supports:
        segment = support.get("segment") or {}
        end_index = segment.get("end_index")
        if not isinstance(end_index, int) or end_index < 0 or end_index > len(text):
            continue
        indices = support.get("grounding_chunk_indices") or []
        citation_links: list[str] = []
        for idx in indices:
            if not isinstance(idx, int):
                continue
            citation_id = chunk_to_id.get(idx)
            if citation_id is None or idx >= len(chunks):
                continue
            web = chunks[idx].get("web") if isinstance(chunks[idx], dict) else {}
            uri = web.get("uri") if isinstance(web, dict) else None
            if uri:
                citation_links.append(f"[{citation_id}]({uri})")

        if citation_links:
            citation_string = ", ".join(citation_links)
            text = text[:end_index] + citation_string + text[end_index:]

    return text, citations


def _collect_citations(candidate: dict) -> tuple[dict[int, str], dict[int, int]]:
    """Collect citation ids from a candidate dict."""
    supports = candidate["grounding_metadata"]["grounding_supports"]
    chunks = candidate["grounding_metadata"]["grounding_chunks"]

    citations: dict[int, str] = {}
    chunk_to_id: dict[int, int] = {}
    next_id = 1

    def label_for(chunk: dict) -> str:
        web = chunk.get("web") or {}
        title = web.get("title")
        uri = web.get("uri")
        if title:
            return title
        if uri:
            parsed = urlparse(uri)
            return parsed.hostname or parsed.netloc or uri
        return "unknown source"

    for support in supports:
        if not isinstance(support, dict):
            continue
        for chunk_idx in support.get("grounding_chunk_indices", []):
            if chunk_idx not in chunk_to_id and 0 <= chunk_idx < len(chunks):
                label = label_for(chunks[chunk_idx])
                chunk_to_id[chunk_idx] = next_id
                citations[next_id] = label
                next_id += 1

    return citations, chunk_to_id
