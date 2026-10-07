"""
main.py

FastAPI entrypoint. Thin HTTP layer only — no business logic.

Features:
- Decision layer integration with on-demand RAGAS evaluation.
- /ask accepts optional 'run_evaluation' parameter.
- If run_evaluation=true, runs RAGAS on-demand and passes scores
  to the pipeline.
- Decision layer then accepts/retries/fallbacks based on quality scores.
- Responses include decision_metadata showing why an answer was
  accepted/rejected.
- BM25 retrieval is rebuilt from the complete Qdrant session corpus
  so multi-document sessions and cold starts use the same knowledge base.
- Upload authorization is checked BEFORE file processing.
"""

import json
import math
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from configs.settings import settings
from ingestion.loader import load_document
from ingestion.chunking import chunk_documents
from vectorstore.vectordb import (
    add_to_vectorstore,
    create_vectorstore,
    delete_vectorstore,
    get_all_documents,
    load_vectorstore,
)
from retrieval.retriever import HybridRetriever
from rag.pipeline import RAGSession, RAGResponse, run_pipeline
from evaluation.ragas_eval import run_ragas_evaluation
from observability.logger import get_logger
from auth import (
    init_auth_db,
    get_current_user,
    CurrentUser,
)


logger = get_logger("main")

init_auth_db()


def _sanitize_floats(obj: Any) -> Any:
    """
    Recursively replace NaN/Inf with None so JSON serialization never crashes.
    """

    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj

    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]

    return obj


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Production RAG API",
    version="2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Persistence helpers ────────────────────────────────────────────────────────

SESSIONS_FILE = settings.STORAGE_DIR / "sessions.json"
EVAL_HISTORY_FILE = settings.STORAGE_DIR / "eval_history.json"


def _load_sessions() -> Dict[str, Any]:
    """Load all chat sessions from disk."""

    if not SESSIONS_FILE.exists():
        return {}

    try:
        return json.loads(
            SESSIONS_FILE.read_text(
                encoding="utf-8-sig"
            )
        ) or {}

    except Exception as e:
        logger.error(
            f"Failed to load sessions: {e}"
        )
        return {}


def _save_sessions(
    sessions: Dict[str, Any]
) -> None:
    """Save all chat sessions to disk."""

    try:
        SESSIONS_FILE.write_text(
            json.dumps(
                sessions,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    except Exception as e:
        logger.error(
            f"Failed to save sessions: {e}"
        )


def _load_eval_history() -> List[Dict[str, Any]]:
    """Load evaluation history from disk."""

    if not EVAL_HISTORY_FILE.exists():
        return []

    try:
        return json.loads(
            EVAL_HISTORY_FILE.read_text(
                encoding="utf-8-sig"
            )
        ) or []

    except Exception as e:
        logger.error(
            f"Failed to load eval history: {e}"
        )
        return []


def _save_eval_history(
    history: List[Dict[str, Any]]
) -> None:
    """Save evaluation history to disk."""

    try:
        EVAL_HISTORY_FILE.write_text(
            json.dumps(
                history,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    except Exception as e:
        logger.error(
            f"Failed to save eval history: {e}"
        )


# ── Global session storage ─────────────────────────────────────────────────────

rag_sessions = {}
chat_sessions = _load_sessions()


def _rebuild_session(
    session_id: str
) -> None:
    """
    Rebuild a complete RAGSession from the persisted Qdrant collection.

    Qdrant is the source of truth for the session's document corpus.
    The complete corpus is reconstructed and supplied to BM25 so that
    retrieval after a restart behaves the same as retrieval before restart.
    """

    if session_id not in chat_sessions:
        return

    vectorstore = load_vectorstore(
        session_id
    )

    if vectorstore is None:
        logger.warning(
            f"Could not load vectorstore for session {session_id}"
        )
        return

    try:
        all_chunks = get_all_documents(
            session_id
        )

        if not all_chunks:
            logger.warning(
                f"No stored documents found for session {session_id}"
            )
            return

        retriever = HybridRetriever(
            vectorstore=vectorstore,
            all_chunks=all_chunks,
        )

        rag_sessions[session_id] = RAGSession(
            session_id=session_id,
            retriever=retriever,
            all_chunks=all_chunks,
        )

        logger.info(
            f"Rebuilt session {session_id} with "
            f"{len(all_chunks)} chunks"
        )

    except Exception as e:
        logger.error(
            f"Failed to rebuild session {session_id}: {e}"
        )


# Load all sessions on startup
for sid in chat_sessions:
    _rebuild_session(sid)


# ── Request/Response Models ────────────────────────────────────────────────────


class QuestionRequest(BaseModel):
    """Request model for asking a question."""

    session_id: str
    question: str
    run_evaluation: bool = False


class SourceInfo(BaseModel):
    """Information about a retrieved source chunk."""

    chunk_id: Optional[str] = None
    source: Optional[str] = None
    page: Optional[int] = None
    chunk_text: Optional[str] = None


class DecisionInfo(BaseModel):
    """Decision layer metadata."""

    decision_type: str
    attempt: int
    max_attempts: int
    reason: str
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    thresholds: Optional[Dict[str, float]] = None


class ChatResponse(BaseModel):
    """Response model for chat/question endpoint."""

    answer: str
    is_grounded: bool
    grounding_score: float
    refused: bool
    sources_count: int
    sources: List[SourceInfo]
    trace_id: str
    timestamp: str
    decision_metadata: Optional[DecisionInfo] = None


class EvaluateRequest(BaseModel):
    """Request model for the optional RAGAS evaluation endpoint."""

    session_id: str
    trace_id: str


class EvaluateResponse(BaseModel):
    """Response model for the optional RAGAS evaluation endpoint."""

    trace_id: str
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    hallucination_rate: Optional[float] = None
    cached: bool


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint."""

    return {
        "status": "ok",
        "sessions_loaded": len(rag_sessions),
    }


# ── Auth ───────────────────────────────────────────────────────────────────────
# Registration and login happen client-side via the Firebase JS SDK.
# The backend verifies the Firebase ID token on every protected request.


@app.post("/upload")
def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    current_user: CurrentUser = Depends(
        get_current_user
    ),
) -> Dict[str, Any]:
    """
    Upload a PDF or DOCX document.

    Creates a new session if session_id is not provided.

    IMPORTANT:
        Authorization for an existing session happens BEFORE the uploaded
        file is read or processed.
    """

    if not file:
        raise HTTPException(
            400,
            "No file provided",
        )

    # ── Generate / determine session_id ─────────────────────────────────────

    if not session_id:
        session_id = str(uuid.uuid4())

    is_new = session_id not in chat_sessions

    # ── AUTHORIZE BEFORE FILE PROCESSING ─────────────────────────────────────
    #
    # Existing sessions must belong to the authenticated user.
    #
    # This check intentionally happens BEFORE:
    #   - reading the file
    #   - loading the document
    #   - chunking
    #   - embedding
    #   - writing to Qdrant
    #
    # This closes P0-004.

    if not is_new:
        if (
            chat_sessions[session_id].get("user_id")
            != current_user.user_id
        ):
            raise HTTPException(
                403,
                "This session belongs to another user",
            )

    # ── Validate file type ──────────────────────────────────────────────────

    file_ext = Path(
        file.filename
    ).suffix.lower()

    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"File type {file_ext} not allowed. "
            f"Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    tmp_path = None

    try:

        # ── Read uploaded file ──────────────────────────────────────────────

        file_content = file.file.read()

        file_size_mb = (
            len(file_content)
            / (1024 * 1024)
        )

        # ── Validate file size ──────────────────────────────────────────────

        if (
            file_size_mb
            > settings.MAX_FILE_SIZE_MB
        ):
            raise HTTPException(
                413,
                f"File size {file_size_mb:.1f}MB "
                f"exceeds limit "
                f"{settings.MAX_FILE_SIZE_MB}MB",
            )

        # ── Write to temporary file ─────────────────────────────────────────

        with tempfile.NamedTemporaryFile(
            suffix=file_ext,
            delete=False,
        ) as tmp:

            tmp.write(
                file_content
            )

            tmp_path = tmp.name

        # ── Load and chunk document ─────────────────────────────────────────

        docs = load_document(
            tmp_path
        )

        if not docs:
            raise ValueError(
                "Document contains no extractable text"
            )

        chunks = chunk_documents(
            docs
        )

        if not chunks:
            raise ValueError(
                "No valid chunks after processing"
            )

        logger.info(
            f"Loaded {len(docs)} doc(s), "
            f"created {len(chunks)} chunk(s) "
            f"for session {session_id}"
        )

        # ── Create or add to vectorstore ───────────────────────────────────

        if is_new:

            vectorstore = create_vectorstore(
                session_id,
                chunks,
            )

        else:

            vectorstore = add_to_vectorstore(
                session_id,
                chunks,
            )

        # ── Reconstruct COMPLETE session corpus ─────────────────────────────
        #
        # `chunks` contains only the currently uploaded document.
        #
        # Qdrant contains every document belonging to this session.
        # Therefore we reconstruct the complete corpus for BM25.

        all_chunks = get_all_documents(
            session_id
        )

        if not all_chunks:
            raise ValueError(
                "No stored chunks found after vectorstore update"
            )

        logger.info(
            f"Session {session_id} now contains "
            f"{len(all_chunks)} total chunks"
        )

        # ── Create RAGSession ───────────────────────────────────────────────

        retriever = HybridRetriever(
            vectorstore=vectorstore,
            all_chunks=all_chunks,
        )

        rag_sessions[session_id] = RAGSession(
            session_id=session_id,
            retriever=retriever,
            all_chunks=all_chunks,
        )

        # ── Initialize / update chat session ────────────────────────────────

        timestamp = datetime.now().isoformat()

        if is_new:

            chat_sessions[session_id] = {
                "session_id": session_id,
                "user_id": current_user.user_id,
                "created_at": timestamp,
                "last_updated": timestamp,
                "filename": file.filename,
                "filenames": [
                    file.filename
                ],
                "file_size_mb": round(
                    file_size_mb,
                    2,
                ),
                "document_count": len(
                    docs
                ),
                "chunk_count": len(
                    chunks
                ),
                "messages": [],
            }

        else:

            existing = chat_sessions[
                session_id
            ]

            existing["document_count"] = (
                existing.get(
                    "document_count",
                    0,
                )
                + len(docs)
            )

            existing["chunk_count"] = (
                existing.get(
                    "chunk_count",
                    0,
                )
                + len(chunks)
            )

            existing.setdefault(
                "filenames",
                (
                    [existing.get("filename")]
                    if existing.get("filename")
                    else []
                ),
            )

            existing["filenames"].append(
                file.filename
            )

            existing["last_updated"] = timestamp

        _save_sessions(
            chat_sessions
        )

        return {
            "session_id": session_id,
            "filename": file.filename,
            "file_size_mb": round(
                file_size_mb,
                2,
            ),
            "document_count": len(
                docs
            ),
            "chunk_count": len(
                chunks
            ),
            "created_at": timestamp,
            "is_new_session": is_new,
        }

    except HTTPException:
        raise

    except Exception as e:

        logger.error(
            f"Upload error: {e}"
        )

        raise HTTPException(
            500,
            f"Upload failed: {e}",
        )

    finally:

        if tmp_path:

            Path(
                tmp_path
            ).unlink(
                missing_ok=True
            )


@app.post(
    "/ask",
    response_model=ChatResponse,
)
def ask_question(
    request: QuestionRequest,
    current_user: CurrentUser = Depends(
        get_current_user
    ),
) -> ChatResponse:
    """
    Ask a question about uploaded documents.

    Supports optional on-demand RAGAS evaluation.

    If run_evaluation=true:
        1. Generate an initial answer.
        2. Evaluate it with RAGAS.
        3. Pass evaluation scores to the decision layer.
        4. Re-run the pipeline with the evaluation scores.
    """

    if not request.question.strip():
        raise HTTPException(
            400,
            "Question cannot be empty",
        )

    if request.session_id not in chat_sessions:
        raise HTTPException(
            404,
            "Session not found",
        )

    if (
        chat_sessions[
            request.session_id
        ].get("user_id")
        != current_user.user_id
    ):
        raise HTTPException(
            403,
            "This session belongs to another user",
        )

    if request.session_id not in rag_sessions:

        _rebuild_session(
            request.session_id
        )

        if request.session_id not in rag_sessions:
            raise HTTPException(
                404,
                "Session not found",
            )

    try:

        session = rag_sessions[
            request.session_id
        ]

        trace_id = str(
            uuid.uuid4()
        )

        # ── Run pipeline ────────────────────────────────────────────────────

        result: RAGResponse = run_pipeline(
            session,
            request.question,
            trace_id=trace_id,
        )

        timestamp = datetime.now().isoformat()

        # ── Optional RAGAS evaluation ───────────────────────────────────────

        eval_scores = None

        if (
            request.run_evaluation
            and not result.refused
        ):

            logger.info(
                f"Running on-demand RAGAS evaluation "
                f"for trace {result.trace_id}"
            )

            contexts = []

            if result.sources:

                for source in result.sources:

                    chunk_text = (
                        source.get(
                            "chunk_text",
                            "",
                        ).strip()
                    )

                    if chunk_text:
                        contexts.append(
                            chunk_text
                        )

            if contexts:

                try:

                    test_case = {
                        "question": request.question,
                        "answer": result.answer,
                        "contexts": contexts,
                    }

                    eval_result = (
                        run_ragas_evaluation(
                            [test_case]
                        )
                    )

                    if (
                        eval_result.get(
                            "error"
                        )
                        or eval_result.get(
                            "evaluated_cases",
                            0,
                        ) == 0
                    ):

                        logger.warning(
                            "RAGAS evaluation had issues: "
                            f"{eval_result.get('error')}"
                        )

                    else:

                        eval_scores = {
                            "faithfulness": (
                                eval_result.get(
                                    "faithfulness",
                                    0.0,
                                )
                            ),
                            "answer_relevancy": (
                                eval_result.get(
                                    "answer_relevancy",
                                    0.0,
                                )
                            ),
                        }

                        logger.info(
                            "RAGAS scores: "
                            f"faithfulness="
                            f"{eval_scores['faithfulness']:.2f}, "
                            f"relevancy="
                            f"{eval_scores['answer_relevancy']:.2f}"
                        )

                        result = run_pipeline(
                            session,
                            request.question,
                            eval_scores=eval_scores,
                            trace_id=trace_id,
                        )

                except Exception as e:

                    logger.error(
                        f"RAGAS evaluation failed: {e}"
                    )

                    logger.info(
                        "Continuing without evaluation"
                    )

            else:

                logger.warning(
                    "No contexts available for RAGAS evaluation"
                )

        # ── Extract contexts for storage ────────────────────────────────────

        contexts = []

        if (
            not result.refused
            and result.sources
        ):

            for source in result.sources:

                chunk_text = (
                    source.get(
                        "chunk_text",
                        "",
                    ).strip()
                )

                source_name = (
                    source.get(
                        "source",
                        "",
                    ).strip()
                )

                if chunk_text or source_name:

                    contexts.append(
                        chunk_text
                        if chunk_text
                        else source_name
                    )

        # ── Ensure message list exists ───────────────────────────────────────

        if request.session_id not in chat_sessions:

            logger.warning(
                f"Session {request.session_id} "
                "disappeared during ask"
            )

            raise HTTPException(
                404,
                "Session not found",
            )

        if (
            "messages"
            not in chat_sessions[
                request.session_id
            ]
        ):

            chat_sessions[
                request.session_id
            ]["messages"] = []

        # ── Store message in history ─────────────────────────────────────────

        message_entry = {
            "question": request.question,
            "answer": result.answer,
            "grounding_score": result.grounding_score,
            "refused": result.refused,
            "trace_id": result.trace_id,
            "timestamp": timestamp,
            "contexts": contexts,
        }

        if eval_scores:

            message_entry[
                "eval_scores"
            ] = eval_scores

        if result.decision_metadata:

            message_entry[
                "decision_metadata"
            ] = result.decision_metadata

        chat_sessions[
            request.session_id
        ]["messages"].append(
            message_entry
        )

        chat_sessions[
            request.session_id
        ]["last_updated"] = timestamp

        _save_sessions(
            chat_sessions
        )

        # ── Build response ───────────────────────────────────────────────────

        decision_info = None

        if result.decision_metadata:

            decision_info = DecisionInfo(
                decision_type=(
                    result.decision_metadata.get(
                        "decision_type"
                    )
                ),
                attempt=(
                    result.decision_metadata.get(
                        "attempt",
                        1,
                    )
                ),
                max_attempts=(
                    result.decision_metadata.get(
                        "max_attempts",
                        1,
                    )
                ),
                reason=(
                    result.decision_metadata.get(
                        "reason",
                        "",
                    )
                ),
                faithfulness=(
                    result.decision_metadata.get(
                        "scores",
                        {},
                    ).get(
                        "faithfulness"
                    )
                ),
                answer_relevancy=(
                    result.decision_metadata.get(
                        "scores",
                        {},
                    ).get(
                        "answer_relevancy"
                    )
                ),
                thresholds=(
                    result.decision_metadata.get(
                        "thresholds"
                    )
                ),
            )

        return ChatResponse(
            answer=result.answer,
            is_grounded=result.is_grounded,
            grounding_score=result.grounding_score,
            refused=result.refused,
            sources_count=len(
                result.sources
            ),
            sources=[
                SourceInfo(**source)
                for source in result.sources
            ],
            trace_id=result.trace_id,
            timestamp=timestamp,
            decision_metadata=decision_info,
        )

    except HTTPException:
        raise

    except Exception as e:

        err = str(e)

        logger.error(
            f"Ask error: {e}"
        )

        if (
            "401" in err
            or "Unauthorized" in err
        ):
            raise HTTPException(
                401,
                "API key invalid",
            )

        if (
            "429" in err
            or "rate_limit" in err.lower()
            or "rate limit" in err.lower()
        ):
            raise HTTPException(
                429,
                "Rate limit exceeded",
            )

        if "timeout" in err.lower():

            raise HTTPException(
                504,
                "Request timeout",
            )

        raise HTTPException(
            500,
            f"Error generating answer: {err}",
        )


@app.post(
    "/evaluate",
    response_model=EvaluateResponse,
)
def evaluate_answer(
    request: EvaluateRequest,
    current_user: CurrentUser = Depends(
        get_current_user
    ),
) -> EvaluateResponse:
    """
    Run an optional, on-demand RAGAS evaluation for a previously generated
    answer.

    Results are cached in eval_history.json by trace_id.
    """

    if request.session_id not in chat_sessions:

        raise HTTPException(
            404,
            "Session not found",
        )

    if (
        chat_sessions[
            request.session_id
        ].get("user_id")
        != current_user.user_id
    ):

        raise HTTPException(
            403,
            "This session belongs to another user",
        )

    # ── Check cache ──────────────────────────────────────────────────────────

    eval_history = _load_eval_history()

    cached = next(
        (
            entry
            for entry in eval_history
            if entry.get("trace_id")
            == request.trace_id
        ),
        None,
    )

    if cached:

        return EvaluateResponse(
            trace_id=request.trace_id,
            faithfulness=cached[
                "faithfulness"
            ],
            answer_relevancy=cached[
                "answer_relevancy"
            ],
            hallucination_rate=cached[
                "hallucination_rate"
            ],
            cached=True,
        )

    # ── Locate original message ──────────────────────────────────────────────

    messages = chat_sessions[
        request.session_id
    ].get(
        "messages",
        [],
    )

    message = next(
        (
            item
            for item in messages
            if item.get("trace_id")
            == request.trace_id
        ),
        None,
    )

    if message is None:

        raise HTTPException(
            404,
            "No stored answer found for this trace_id",
        )

    if message.get("refused"):

        raise HTTPException(
            400,
            "Cannot evaluate a refused/ungrounded "
            "answer — no answer was generated",
        )

    contexts = message.get(
        "contexts"
    ) or []

    if not contexts:

        raise HTTPException(
            400,
            "No retrieved context available to evaluate against",
        )

    # ── Run RAGAS ────────────────────────────────────────────────────────────

    test_case = {
        "question": message["question"],
        "answer": message["answer"],
        "contexts": contexts,
    }

    try:

        result = run_ragas_evaluation(
            [test_case]
        )

    except Exception as e:

        logger.error(
            f"RAGAS evaluation failed "
            f"for trace {request.trace_id}: {e}"
        )

        raise HTTPException(
            500,
            f"Evaluation failed: {e}",
        )

    if (
        result.get("error")
        or result.get(
            "evaluated_cases",
            0,
        ) == 0
    ):

        raise HTTPException(
            500,
            "Evaluation failed: "
            f"{result.get('error', 'no valid cases evaluated')}",
        )

    result = _sanitize_floats(
        result
    )

    response = EvaluateResponse(
        trace_id=request.trace_id,
        faithfulness=result[
            "faithfulness"
        ],
        answer_relevancy=result[
            "answer_relevancy"
        ],
        hallucination_rate=result[
            "hallucination_rate"
        ],
        cached=False,
    )

    # ── Persist evaluation ───────────────────────────────────────────────────

    eval_history.append(
        {
            "trace_id": request.trace_id,
            "session_id": request.session_id,
            "faithfulness": response.faithfulness,
            "answer_relevancy": response.answer_relevancy,
            "hallucination_rate": response.hallucination_rate,
            "evaluated_at": datetime.now().isoformat(),
        }
    )

    _save_eval_history(
        eval_history
    )

    return response


@app.get("/sessions")
def list_sessions(
    current_user: CurrentUser = Depends(
        get_current_user
    ),
) -> Dict[str, Any]:
    """List sessions belonging to the current user."""

    return {
        "sessions": [
            {
                "session_id": sid,
                "created_at": sess.get(
                    "created_at"
                ),
                "last_updated": sess.get(
                    "last_updated"
                ),
                "filename": sess.get(
                    "filename"
                ),
                "message_count": len(
                    sess.get(
                        "messages",
                        [],
                    )
                ),
                "document_count": sess.get(
                    "document_count"
                ),
                "chunk_count": sess.get(
                    "chunk_count"
                ),
            }
            for sid, sess in chat_sessions.items()
            if sess.get("user_id")
            == current_user.user_id
        ]
    }


@app.get(
    "/session/{session_id}"
)
def get_session(
    session_id: str,
    current_user: CurrentUser = Depends(
        get_current_user
    ),
) -> Dict[str, Any]:
    """Get session details and full chat history."""

    if session_id not in chat_sessions:

        raise HTTPException(
            404,
            "Session not found",
        )

    if (
        chat_sessions[
            session_id
        ].get("user_id")
        != current_user.user_id
    ):

        raise HTTPException(
            403,
            "This session belongs to another user",
        )

    session = chat_sessions[
        session_id
    ]

    return {
        "session_id": session_id,
        "created_at": session.get(
            "created_at"
        ),
        "last_updated": session.get(
            "last_updated"
        ),
        "filename": session.get(
            "filename"
        ),
        "document_count": session.get(
            "document_count"
        ),
        "chunk_count": session.get(
            "chunk_count"
        ),
        "messages": session.get(
            "messages",
            [],
        ),
    }


@app.delete(
    "/session/{session_id}"
)
def delete_session(
    session_id: str,
    current_user: CurrentUser = Depends(
        get_current_user
    ),
) -> Dict[str, str]:
    """Delete a session and its associated vectorstore."""

    if session_id not in chat_sessions:

        raise HTTPException(
            404,
            "Session not found",
        )

    if (
        chat_sessions[
            session_id
        ].get("user_id")
        != current_user.user_id
    ):

        raise HTTPException(
            403,
            "This session belongs to another user",
        )

    try:

        delete_vectorstore(
            session_id
        )

        del chat_sessions[
            session_id
        ]

        if session_id in rag_sessions:

            del rag_sessions[
                session_id
            ]

        _save_sessions(
            chat_sessions
        )

        return {
            "message": (
                f"Session {session_id} deleted"
            )
        }

    except Exception as e:

        logger.error(
            f"Delete session error: {e}"
        )

        raise HTTPException(
            500,
            f"Failed to delete session: {e}",
        )


if __name__ == "__main__":

    import os

    port = int(
        os.environ.get(
            "PORT",
            8000,
        )
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )