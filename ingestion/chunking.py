"""
ingestion/chunking.py

Semantic-aware chunking strategy for production RAG.

Why the naive approach breaks:
  1. Fixed character/token windows split mid-sentence → incomplete context
  2. 1024-char chunks with 100-char overlap is ~10% overlap — too little for
     dense technical text where a single paragraph can span multiple chunks.
  3. No minimum length filter → boilerplate headers/footers become chunks and
     pollute retrieval with noise.
  4. Metadata like chunk position (section, paragraph index) is discarded,
     making it impossible to reconstruct surrounding context.

Our strategy:
  - RecursiveCharacterTextSplitter with semantic separators (headings → paragraphs
    → sentences → words → chars) — splits at the most natural boundary available.
  - Token-based sizing using tiktoken so chunk_size is comparable across models.
  - Minimum-length filter removes degenerate chunks (headers, page numbers, etc.).
  - Each chunk gets a deterministic chunk_id, position index, and parent metadata.
"""

import hashlib
import uuid
from typing import List

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from configs.settings import settings
from observability.logger import get_logger

logger = get_logger("ingestion.chunking")

# Tiktoken encoder for token counting (model-agnostic cl100k_base)
_ENCODER = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_ENCODER.encode(text))


def _chunk_id(source: str, index: int, content: str) -> str:
    """Stable chunk ID based on source + index + first 64 chars of content."""
    fingerprint = f"{source}::{index}::{content[:64]}"
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:12]


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split a list of loaded Documents into production-quality chunks.

    Returns chunks with enriched metadata:
        chunk_id       — stable deterministic ID
        chunk_index    — position within the source document
        chunk_total    — total chunk count for this source
        token_count    — number of tokens in this chunk
        source / page  — inherited from parent document
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=_token_len,          # ← token-based, not char-based
        separators=[
            "\n## ", "\n### ", "\n#### ",   # markdown headings
            "\n\n",                          # paragraph breaks
            "\n",                            # line breaks
            ". ", "? ", "! ",               # sentence boundaries
            " ", "",                         # last resort: words → chars
        ],
        is_separator_regex=False,
        add_start_index=True,               # records char offset in metadata
    )

    all_chunks: List[Document] = []

    for doc in documents:
        raw_chunks = splitter.split_documents([doc])

        # Filter degenerate chunks
        valid_chunks = [
            c for c in raw_chunks
            if len(c.page_content.strip()) >= settings.CHUNK_MIN_CHARS
        ]

        source = doc.metadata.get("source", "unknown")
        total  = len(valid_chunks)

        for i, chunk in enumerate(valid_chunks):
            tokens = _token_len(chunk.page_content)
            
            # Sanitize metadata to only str, int, float, bool (required by vector store payload)
            safe_metadata = {}
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    safe_metadata[k] = v
                elif v is not None:
                    safe_metadata[k] = str(v)
            chunk.metadata = safe_metadata

            chunk.metadata.update({
                "chunk_id":    _chunk_id(source, i, chunk.page_content),
                "chunk_index": i,
                "chunk_total": total,
                "token_count": tokens,
            })
            all_chunks.append(chunk)

    logger.info(
        f"Chunked {len(documents)} document(s) → {len(all_chunks)} chunk(s) "
        f"(avg {sum(_token_len(c.page_content) for c in all_chunks) // max(len(all_chunks),1)} tokens/chunk)"
    )
    return all_chunks