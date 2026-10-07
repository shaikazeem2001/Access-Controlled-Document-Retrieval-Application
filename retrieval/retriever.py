"""
retrieval/retriever.py

Hybrid dense + BM25 retrieval with RRF
(Reciprocal Rank Fusion).

The retriever receives an explicit RetrievalConfig so
the Pipeline and Decision Layer can control retrieval
without relying on hard-coded settings.
"""

from typing import List, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from configs.settings import settings
from observability.logger import get_logger
from retrieval.config import RetrievalConfig


logger = get_logger("retrieval.retriever")


def _reciprocal_rank_fusion(
    dense_docs: List[Document],
    sparse_docs: List[Document],
    alpha: float = settings.HYBRID_ALPHA,
    k: int = 60,
) -> List[Tuple[Document, float]]:
    """
    Combine dense and sparse retrieval results using
    Reciprocal Rank Fusion.

    Args:
        dense_docs:
            Documents returned by dense retrieval.

        sparse_docs:
            Documents returned by BM25.

        alpha:
            Weight assigned to dense retrieval.

        k:
            RRF normalization constant.

    Returns:
        List of (Document, fusion_score) tuples sorted
        by descending fusion score.
    """

    scores = {}
    doc_map = {}

    def _key(doc: Document) -> str:
        return doc.metadata.get(
            "chunk_id",
            doc.page_content[:64],
        )

    # Dense results
    for rank, doc in enumerate(dense_docs):
        key = _key(doc)

        scores[key] = (
            scores.get(key, 0.0)
            + alpha * (1.0 / (k + rank + 1))
        )

        doc_map[key] = doc

    # Sparse results
    for rank, doc in enumerate(sparse_docs):
        key = _key(doc)

        scores[key] = (
            scores.get(key, 0.0)
            + (1.0 - alpha)
            * (1.0 / (k + rank + 1))
        )

        doc_map[key] = doc

    sorted_keys = sorted(
        scores,
        key=lambda key: scores[key],
        reverse=True,
    )

    return [
        (doc_map[key], scores[key])
        for key in sorted_keys
    ]


class HybridRetriever:
    """
    Hybrid retriever combining:

        Dense retrieval
             +
        BM25 retrieval
             ↓
        RRF fusion

    Retrieval parameters are supplied through RetrievalConfig.
    """

    def __init__(
        self,
        vectorstore: QdrantVectorStore,
        all_chunks: List[Document],
    ) -> None:
        """
        Initialize the hybrid retriever.

        Args:
            vectorstore:
                Qdrant vector store used for dense retrieval.

            all_chunks:
                All documents belonging to the session.
                Used to construct the BM25 index.
        """

        self._vectorstore = vectorstore

        # Build BM25 with enough capacity for the largest
        # retry request. Individual retrieval calls then
        # slice the result to config.top_k_sparse.
        max_sparse_k = max(
            settings.TOP_K_SPARSE,
            settings.TOP_K_SPARSE
            + (
                settings.RETRY_RETRIEVAL_TOP_K_INCREASE
                * max(
                    1,
                    settings.RAGAS_MAX_RETRY_ATTEMPTS,
                )
            ),
        )

        self._bm25 = BM25Retriever.from_documents(
            all_chunks,
            k=max_sparse_k,
        )

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig,
    ) -> Tuple[List[Document], List[float]]:
        """
        Retrieve documents using the supplied retrieval configuration.

        Args:
            query:
                User question or rewritten search query.

            config:
                RetrievalConfig controlling dense top-k,
                sparse top-k, and dense score threshold.

        Returns:
            Tuple of:

                documents:
                    Documents ordered by RRF score.

                scores:
                    Corresponding RRF fusion scores.
        """

        logger.info(
            "Retrieval started | "
            f"dense_k={config.top_k_dense} | "
            f"sparse_k={config.top_k_sparse} | "
            f"threshold={config.score_threshold}"
        )

        # ──────────────────────────────────────────────
        # Dense retrieval
        # ──────────────────────────────────────────────

        dense_results = (
            self._vectorstore
            .similarity_search_with_relevance_scores(
                query,
                k=config.top_k_dense,
            )
        )

        top_raw = (
            round(dense_results[0][1], 3)
            if dense_results
            else "n/a"
        )

        logger.info(
            "Dense retrieval: "
            f"{len(dense_results)} docs, "
            f"top score={top_raw}"
        )

        dense_filtered = [
            (doc, score)
            for doc, score in dense_results
            if score >= config.score_threshold
        ]

        if not dense_filtered:
            logger.warning(
                "No dense documents passed "
                f"threshold={config.score_threshold} "
                f"(top={top_raw}) for: "
                f"{query[:80]}"
            )

            dense_docs = []

        else:
            dense_docs = [
                doc
                for doc, _ in dense_filtered
            ]

            logger.info(
                f"{len(dense_docs)} dense docs "
                "passed threshold"
            )

        # ──────────────────────────────────────────────
        # Sparse retrieval
        # ──────────────────────────────────────────────

        # BM25 is initialized with the maximum required k.
        # We explicitly slice here so every call obeys the
        # RetrievalConfig without mutating self._bm25.k.
        sparse_docs = self._bm25.invoke(query)

        sparse_docs = sparse_docs[
            : config.top_k_sparse
        ]

        logger.info(
            "Sparse retrieval: "
            f"{len(sparse_docs)} docs"
        )

        # ──────────────────────────────────────────────
        # RRF fusion
        # ──────────────────────────────────────────────

        fused = _reciprocal_rank_fusion(
            dense_docs,
            sparse_docs,
        )

        docs = [
            doc
            for doc, _ in fused
        ]

        scores = [
            score
            for _, score in fused
        ]

        logger.info(
            "Final hybrid result: "
            f"{len(docs)} docs"
        )

        return docs, scores