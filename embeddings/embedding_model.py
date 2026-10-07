"""
embeddings/embedding_model.py

Centralized embedding model for the RAG system.

The project uses BAAI/bge-small-en-v1.5 through
LangChain's HuggingFaceEmbeddings.

The embedding model is cached in memory with lru_cache
so it is instantiated only once per process.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from configs.settings import settings
from observability.logger import get_logger


logger = get_logger("embeddings.model")


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Return the configured Hugging Face embedding model.

    The function is cached so repeated calls reuse the same
    embedding model instance within the process.

    Returns:
        HuggingFaceEmbeddings:
            Configured embedding model.
    """

    logger.info(
        "Loading embedding model: "
        f"{settings.EMBEDDING_MODEL}"
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={
            "device": settings.EMBEDDING_DEVICE,
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": settings.EMBEDDING_BATCH_SIZE,
        },
    )

    logger.info(
        "Embedding model loaded successfully"
    )

    return embeddings