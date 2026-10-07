"""
observability/logger.py

Structured JSON logging for every RAG step.
Every retrieval, rerank, prompt, and response is recorded with enough
context to reproduce and debug any failure.
"""

import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from configs.settings import settings


# ── Structured JSON formatter ──────────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # attach any extra fields passed via logger.info(..., extra={...})
        for key in ("trace_id", "session_id", "step", "data"):
            if hasattr(record, key):
                log_obj[key] = getattr(record, key)
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def get_logger(name: str) -> logging.Logger:
    """Return a structured JSON logger that writes to file and stderr."""
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))
    fmt = JSONFormatter()

    # stderr handler
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # rotating file handler
    fh = logging.FileHandler(settings.LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── RAGTrace — per-request audit object ───────────────────────────────────────

class RAGTrace:
    """
    Collects every signal from a single RAG request so you can:
    - Debug retrieval misses
    - Inspect context passed to the LLM
    - Audit hallucination risk scores
    - Compare against eval ground truths

    Usage:
        trace = RAGTrace(session_id="abc", question="...")
        trace.set_retrieval(docs, scores)
        trace.set_reranked(reranked_docs)
        trace.set_prompt(filled_prompt)
        trace.set_response(answer, grounding_score)
        trace.finish()         # logs everything
    """

    def __init__(self, session_id: str, question: str):
        self.trace_id   = str(uuid.uuid4())
        self.session_id = session_id
        self.question   = question
        self.started_at = time.perf_counter()
        self._log       = get_logger("rag.trace")

        self.retrieval_docs:   List[Dict] = []
        self.reranked_docs:    List[Dict] = []
        self.prompt:           Optional[str] = None
        self.answer:           Optional[str] = None
        self.grounding_score:  Optional[float] = None
        self.refused:          bool = False
        self.error:            Optional[str] = None

    # -- setters ---------------------------------------------------------------

    def set_retrieval(self, docs: List[Any], scores: List[float]) -> None:
        self.retrieval_docs = [
            {
                "rank": i + 1,
                "score": round(scores[i], 4),
                "source": getattr(docs[i], "metadata", {}).get("source", "unknown"),
                "page":   getattr(docs[i], "metadata", {}).get("page"),
                "chunk_id": getattr(docs[i], "metadata", {}).get("chunk_id"),
                "preview": docs[i].page_content[:200],
            }
            for i in range(len(docs))
        ]

    def set_reranked(self, docs: List[Any]) -> None:
        self.reranked_docs = [
            {
                "rank": i + 1,
                "source": getattr(docs[i], "metadata", {}).get("source", "unknown"),
                "chunk_id": getattr(docs[i], "metadata", {}).get("chunk_id"),
                "preview": docs[i].page_content[:200],
            }
            for i in range(len(docs))
        ]

    def set_prompt(self, prompt: str) -> None:
        self.prompt = prompt

    def set_response(self, answer: str, grounding_score: float) -> None:
        self.answer          = answer
        self.grounding_score = grounding_score

    def set_refused(self) -> None:
        self.refused = True

    def set_error(self, error: str) -> None:
        self.error = error

    # -- finish ----------------------------------------------------------------

    def finish(self) -> Dict[str, Any]:
        elapsed_ms = round((time.perf_counter() - self.started_at) * 1000, 1)

        payload = {
            "trace_id":        self.trace_id,
            "session_id":      self.session_id,
            "question":        self.question,
            "elapsed_ms":      elapsed_ms,
            "retrieval_count": len(self.retrieval_docs),
            "rerank_count":    len(self.reranked_docs),
            "grounding_score": self.grounding_score,
            "refused":         self.refused,
            "error":           self.error,
            "retrieval_docs":  self.retrieval_docs,
            "reranked_docs":   self.reranked_docs,
            # do NOT log the full prompt by default — can contain PII
            "prompt_len":      len(self.prompt) if self.prompt else 0,
            "answer_len":      len(self.answer) if self.answer else 0,
        }

        self._log.info(
            "rag_trace",
            extra={"trace_id": self.trace_id, "session_id": self.session_id,
                   "step": "finish", "data": payload}
        )
        return payload

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id":       self.trace_id,
            "grounding_score": self.grounding_score,
            "refused":        self.refused,
            "retrieval_docs": self.retrieval_docs,
        }