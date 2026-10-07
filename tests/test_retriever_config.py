from unittest.mock import Mock

from langchain_core.documents import Document

from retrieval.config import RetrievalConfig
from retrieval.retriever import HybridRetriever


def test_retriever_uses_supplied_config():
    vectorstore = Mock()

    vectorstore.similarity_search_with_relevance_scores.return_value = [
        (
            Document(
                page_content="document 1",
                metadata={"chunk_id": "1"},
            ),
            0.80,
        ),
        (
            Document(
                page_content="document 2",
                metadata={"chunk_id": "2"},
            ),
            0.20,
        ),
    ]

    retriever = HybridRetriever.__new__(
        HybridRetriever
    )

    retriever._vectorstore = vectorstore

    bm25 = Mock()

    bm25.invoke.return_value = [
        Document(
            page_content="document 1",
            metadata={"chunk_id": "1"},
        ),
        Document(
            page_content="document 3",
            metadata={"chunk_id": "3"},
        ),
        Document(
            page_content="document 4",
            metadata={"chunk_id": "4"},
        ),
    ]

    retriever._bm25 = bm25

    config = RetrievalConfig(
        top_k_dense=2,
        top_k_sparse=1,
        score_threshold=0.50,
    )

    docs, scores = retriever.retrieve(
        "test query",
        config,
    )

    # ──────────────────────────────────────────────
    # Verify dense top-k
    # ──────────────────────────────────────────────

    vectorstore.similarity_search_with_relevance_scores.assert_called_once_with(
        "test query",
        k=2,
    )

    # ──────────────────────────────────────────────
    # Verify sparse retrieval was called
    # ──────────────────────────────────────────────

    bm25.invoke.assert_called_once_with(
        "test query",
    )

    # ──────────────────────────────────────────────
    # Verify sparse top-k
    # ──────────────────────────────────────────────

    # BM25 returned 3 documents, but the retriever
    # must only use the first config.top_k_sparse.
    #
    # The first BM25 document is document 1.
    assert docs[0].metadata["chunk_id"] == "1"

    # ──────────────────────────────────────────────
    # Verify score threshold
    # ──────────────────────────────────────────────

    # document 2 has dense score 0.20 and must be
    # filtered because threshold = 0.50.
    assert all(
        doc.metadata["chunk_id"] != "2"
        for doc in docs
    )

    # ──────────────────────────────────────────────
    # Verify RRF deduplication
    # ──────────────────────────────────────────────

    # document 1 appears in both dense and sparse
    # retrieval, so RRF returns it only once.
    assert len(docs) == 1
    assert len(scores) == 1