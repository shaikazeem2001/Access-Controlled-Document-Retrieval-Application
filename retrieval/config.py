"""
retrieval/config.py

Shared configuration contract for the retrieval layer.

This keeps the Decision Layer, Pipeline, and Retriever
consistent about which parameters control retrieval.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalConfig:
    """
    Configuration used by HybridRetriever.

    Attributes:
        top_k_dense:
            Number of candidates requested from dense retrieval.

        top_k_sparse:
            Number of candidates used from BM25 retrieval.

        score_threshold:
            Minimum dense relevance score required for a
            document to participate in hybrid fusion.
    """

    top_k_dense: int
    top_k_sparse: int
    score_threshold: float

    def __post_init__(self) -> None:
        if self.top_k_dense <= 0:
            raise ValueError(
                "top_k_dense must be greater than 0"
            )

        if self.top_k_sparse <= 0:
            raise ValueError(
                "top_k_sparse must be greater than 0"
            )

        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError(
                "score_threshold must be between 0.0 and 1.0"
            )