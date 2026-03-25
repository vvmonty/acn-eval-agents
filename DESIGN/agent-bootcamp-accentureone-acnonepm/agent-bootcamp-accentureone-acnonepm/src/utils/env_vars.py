"""Environment variable configuration management.

This module provides a centralized interface for loading and accessing
configuration values from environment variables. It uses Pydantic for
type validation and automatic loading from .env files.
"""

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_GEMINI_OPENAI_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)


def _with_https_scheme(url: str) -> str:
    s = url.strip()
    if not s.startswith(("http://", "https://")):
        return f"https://{s.lstrip('/')}"
    return s


class Configs(BaseSettings):
    """Configuration settings loaded from environment variables.

    This class automatically loads configuration values from environment variables
    and a .env file, and provides type-safe access to all settings. It validates
    environment variables on instantiation.

    Attributes
    ----------
    openai_base_url : str
        Base URL for OpenAI-compatible API (defaults to Gemini endpoint).
    openai_api_key : str
        API key for OpenAI-compatible API (accepts OPENAI_API_KEY, GEMINI_API_KEY,
        or GOOGLE_API_KEY).
    default_planner_model : str, default='gemini-2.5-pro'
        Model name for planning tasks. This is typically a more capable and expensive
        model.
    default_worker_model : str, default='gemini-2.5-flash'
        Model name for worker tasks. This is typically a less expensive model.
    embedding_base_url : str or None
        Base URL for embedding API service (optional; required for KB / eval scripts).
    embedding_api_key : str or None
        API key for embedding service (optional; required for KB / eval scripts).
    embedding_model_name : str, default='@cf/baai/bge-m3'
        Name of the embedding model.
    weaviate_collection_name : str, default='enwiki_20250520'
        Name of the Weaviate collection to use.
    weaviate_api_key : str or None
        API key for Weaviate (optional; defaults unset with local fallback in client).
    weaviate_http_host : str or None
        Weaviate HTTP host (optional; e.g. ``*.weaviate.cloud`` or omitted for localhost).
    weaviate_grpc_host : str or None
        Weaviate gRPC host (optional; e.g. ``grpc-*.weaviate.cloud`` or omitted for localhost).
    weaviate_http_port : int, default=443
        Port for Weaviate HTTP connections.
    weaviate_grpc_port : int, default=443
        Port for Weaviate gRPC connections.
    weaviate_http_secure : bool, default=True
        Use secure HTTP connection.
    weaviate_grpc_secure : bool, default=True
        Use secure gRPC connection.
    langfuse_public_key : str
        Langfuse public key (must start with pk-lf-).
    langfuse_secret_key : str
        Langfuse secret key (must start with sk-lf-).
    langfuse_host : str, default='https://us.cloud.langfuse.com'
        Langfuse host URL.
    e2b_api_key : str or None
        Optional E2B.dev API key for code interpreter (must start with e2b_).
    default_code_interpreter_template : str or None
        Optional default template name or ID for E2B.dev code interpreter.
    web_search_base_url : str or None
        Optional base URL for web search service.
    web_search_api_key : str or None
        Optional API key for web search service.

    Examples
    --------
    >>> from src.utils.env_vars import Configs
    >>> config = Configs()
    >>> print(config.default_planner_model)
    'gemini-2.5-pro'

    Notes
    -----
    Create a .env file in your project root with the required environment
    variables. The class will automatically load and validate them.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    openai_base_url: str = DEFAULT_GEMINI_OPENAI_BASE_URL
    openai_api_key: str = Field(
        validation_alias=AliasChoices(
            "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"
        )
    )

    @field_validator("openai_base_url", mode="before")
    @classmethod
    def _normalize_openai_base_url(cls, value: object) -> str:
        if value is None or (isinstance(value, str) and not value.strip()):
            return DEFAULT_GEMINI_OPENAI_BASE_URL
        if not isinstance(value, str):
            return DEFAULT_GEMINI_OPENAI_BASE_URL
        return _with_https_scheme(value)

    @field_validator("embedding_base_url", mode="before")
    @classmethod
    def _normalize_embedding_base_url(cls, value: object) -> str | None:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        if not isinstance(value, str):
            return None
        return _with_https_scheme(value)

    @field_validator("embedding_api_key", mode="before")
    @classmethod
    def _empty_embedding_api_key(cls, value: object) -> str | None:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return value if isinstance(value, str) else None

    @field_validator(
        "weaviate_api_key", "weaviate_http_host", "weaviate_grpc_host", mode="before"
    )
    @classmethod
    def _empty_weaviate_optional(cls, value: object) -> str | None:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return value if isinstance(value, str) else None

    default_planner_model: str = "gemini-2.5-pro"
    default_worker_model: str = "gemini-2.5-flash"

    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model_name: str = "@cf/baai/bge-m3"

    weaviate_collection_name: str = "enwiki_20250520"
    weaviate_api_key: str | None = None
    weaviate_http_host: str | None = None
    weaviate_grpc_host: str | None = None
    weaviate_http_port: int = 443
    weaviate_grpc_port: int = 443
    weaviate_http_secure: bool = True
    weaviate_grpc_secure: bool = True

    langfuse_public_key: str = Field(pattern=r"^pk-lf-.*$")
    langfuse_secret_key: str = Field(pattern=r"^sk-lf-.*$")
    langfuse_host: str = "https://us.cloud.langfuse.com"

    # Optional E2B.dev API key for Python Code Interpreter tool
    e2b_api_key: str | None = Field(default=None, pattern=r"^e2b_.*$")
    default_code_interpreter_template: str | None = "9p6favrrqijhasgkq1tv"

    # Optional configs for web search tool
    web_search_base_url: str | None = None
    web_search_api_key: str | None = None
