"""
configs/settings.py

Central configuration for the RAG production system.

All environment-dependent configuration is loaded through Pydantic
settings so that application code does not need to read environment
variables directly.
"""

from pathlib import Path
from typing import Set

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────────────────────────────────────
# Base directory
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):

    # ─────────────────────────────────────────────────────────────────────────
    # Application paths
    # ─────────────────────────────────────────────────────────────────────────

    BASE_DIR: Path = BASE_DIR

    STORAGE_DIR: Path = BASE_DIR / "storage"

    VECTORSTORE_DIR: Path = BASE_DIR / "vectorstore"

    LOG_DIR: Path = BASE_DIR / "logs"

    # ─────────────────────────────────────────────────────────────────────────
    # LLM configuration
    # ─────────────────────────────────────────────────────────────────────────

    LLM_API_KEY: str = ""

    LLM_API_BASE: str = "https://api.groq.com/openai/v1"

    LLM_MODEL: str = "llama-3.3-70b-versatile"

    LLM_TEMPERATURE: float = 0.0

    LLM_MAX_TOKENS: int = 1024

    # ─────────────────────────────────────────────────────────────────────────
    # Qdrant
    # ─────────────────────────────────────────────────────────────────────────

    QDRANT_URL: str = ""

    QDRANT_API_KEY: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # JWT / Authentication
    # ─────────────────────────────────────────────────────────────────────────

    JWT_SECRET_KEY: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # RAGAS evaluation
    # ─────────────────────────────────────────────────────────────────────────

    RAGAS_EVAL_MODEL: str = "llama-3.3-70b-versatile"

    RAGAS_API_BASE: str = "https://api.groq.com/openai/v1"

    RAGAS_MAX_TOKENS: int = 1024

    RAGAS_MAX_CONTEXTS: int = 3

    RAGAS_CONTEXT_MAX_CHARS: int = 500

    RAGAS_RETRY_COUNT: int = 2

    RAGAS_TIMEOUT: int = 180

    # ─────────────────────────────────────────────────────────────────────────
    # Embeddings
    # ─────────────────────────────────────────────────────────────────────────

    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    EMBEDDING_BATCH_SIZE: int = 64

    # ─────────────────────────────────────────────────────────────────────────
    # Chunking
    # ─────────────────────────────────────────────────────────────────────────

    CHUNK_SIZE: int = 512

    CHUNK_OVERLAP: int = 64

    CHUNK_MIN_CHARS: int = 100

    # ─────────────────────────────────────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────────────────────────────────────

    TOP_K_DENSE: int = 20

    TOP_K_SPARSE: int = 20

    TOP_K_RERANK: int = 10

    # Dense retrieval confidence threshold.
    #
    # IMPORTANT:
    # This is NOT the claim-level grounding threshold.
    # It is used by the retriever to filter dense-search results.
    RETRIEVAL_SCORE_THRESHOLD: float = 0.35

    # Weight used during dense + sparse hybrid fusion.
    HYBRID_ALPHA: float = 0.7

    # ─────────────────────────────────────────────────────────────────────────
    # Reranking
    # ─────────────────────────────────────────────────────────────────────────

    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    RERANKER_ENABLED: bool = True

    # ─────────────────────────────────────────────────────────────────────────
    # Claim-level grounding
    # ─────────────────────────────────────────────────────────────────────────
    #
    # The old grounding implementation only checked whether the answer's
    # [SOURCE:chunk_id] citations existed in the retrieved documents.
    #
    # That proves citation validity, but it does NOT prove that the claim
    # itself is actually supported by the cited context.
    #
    # The new grounding implementation uses an NLI model:
    #
    #     claim + context
    #            ↓
    #          NLI model
    #            ↓
    #       entailment score
    #
    # A score >= GROUNDING_NLI_THRESHOLD means the context is considered
    # sufficiently supportive of the claim.

    GROUNDING_NLI_MODEL: str = (
        "cross-encoder/nli-deberta-v3-base"
    )

    GROUNDING_NLI_THRESHOLD: float = 0.7

    # Overall grounding threshold.
    #
    # This remains available for the grounding guardrail's final decision.
    # The claim-level implementation should use the NLI threshold above
    # when determining whether individual claims are supported.
    GROUNDING_MIN_SCORE: float = 0.5

    # ─────────────────────────────────────────────────────────────────────────
    # Refusal
    # ─────────────────────────────────────────────────────────────────────────

    REFUSAL_MESSAGE: str = (
        "I cannot answer this based on the provided documents."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────────────

    LOG_LEVEL: str = "INFO"

    LOG_FILE: Path = LOG_DIR / "rag.log"

    # ─────────────────────────────────────────────────────────────────────────
    # LangSmith
    # ─────────────────────────────────────────────────────────────────────────

    LANGSMITH_API_KEY: str | None = None

    LANGSMITH_PROJECT: str | None = None

    LANGCHAIN_TRACING: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # File ingestion
    # ─────────────────────────────────────────────────────────────────────────

    MAX_FILE_SIZE_MB: int = 50

    ALLOWED_EXTENSIONS: Set[str] = {
        ".txt",
        ".md",
        ".pdf",
        ".docx",
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Query rewriting
    # ─────────────────────────────────────────────────────────────────────────

    ENABLE_QUERY_REWRITING: bool = True

    QUERY_AMBIGUITY_THRESHOLD: float = 0.4

    QUERY_MIN_LENGTH: int = 5

    QUERY_MAX_REWRITE_LENGTH: int = 150

    QUERY_REWRITE_MODEL: str = "openai/gpt-oss-120b"

    # ─────────────────────────────────────────────────────────────────────────
    # RAGAS decision layer
    # ─────────────────────────────────────────────────────────────────────────

    RAGAS_FAITHFULNESS_THRESHOLD: float = 0.7

    RAGAS_RELEVANCE_THRESHOLD: float = 0.65

    RAGAS_MAX_RETRY_ATTEMPTS: int = 2

    RAGAS_RETRY_MODE: str = "better_retrieval"

    RAGAS_FALLBACK_ENABLED: bool = True

    # Increase retrieval breadth during RETRY.
    RETRY_RETRIEVAL_TOP_K_INCREASE: int = 10

    # Lower dense retrieval threshold during RETRY.
    RETRY_LOWERED_THRESHOLD: float = 0.3

    # Increase maximum context size during RETRY.
    RETRY_CONTEXT_MAX_CHARS_INCREASE: int = 1000

    # Optional fallback model.
    FALLBACK_LLM_MODEL: str | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # OpenRouter
    # ─────────────────────────────────────────────────────────────────────────

    openrouter_api_key: str | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # Pydantic settings configuration
    # ─────────────────────────────────────────────────────────────────────────

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Global settings instance
# ─────────────────────────────────────────────────────────────────────────────

settings = Settings()


# ─────────────────────────────────────────────────────────────────────────────
# Ensure required directories exist
# ─────────────────────────────────────────────────────────────────────────────

settings.STORAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

settings.VECTORSTORE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

settings.LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)