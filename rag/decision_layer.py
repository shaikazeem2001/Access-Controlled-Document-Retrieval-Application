
"""
rag/decision_layer.py

RAGAS score-based decision layer for DocMind RAG system.

Makes decisions:

    ACCEPT
    RETRY
    FALLBACK
    REJECT

The RETRY configuration uses the shared RetrievalConfig
contract from the retrieval layer.
"""

from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
import uuid

from configs.settings import settings
from observability.logger import get_logger
from retrieval.config import RetrievalConfig


logger = get_logger("rag.decision_layer")


class DecisionLayer:
    """
    Decide whether the current answer should be:

        ACCEPT
        RETRY
        FALLBACK
        REJECT

    RETRY means:
        Broaden retrieval and generate a new answer.

    FALLBACK means:
        Use the configured fallback LLM.

    REJECT means:
        Maximum retry attempts have been exceeded.
    """

    def __init__(self):
        """Initialize decision thresholds from settings."""

        self.faithfulness_threshold = (
            settings.RAGAS_FAITHFULNESS_THRESHOLD
        )

        self.relevance_threshold = (
            settings.RAGAS_RELEVANCE_THRESHOLD
        )

        self.max_retry_attempts = (
            settings.RAGAS_MAX_RETRY_ATTEMPTS
        )

        self.retry_mode = (
            settings.RAGAS_RETRY_MODE
        )

        self.fallback_enabled = (
            settings.RAGAS_FALLBACK_ENABLED
        )

        logger.info(
            "DecisionLayer initialized: "
            f"faithfulness_threshold="
            f"{self.faithfulness_threshold}, "
            f"relevance_threshold="
            f"{self.relevance_threshold}, "
            f"max_retries="
            f"{self.max_retry_attempts}"
        )

    # ──────────────────────────────────────────────
    # SCORE EVALUATION
    # ──────────────────────────────────────────────

    def should_retry(
        self,
        scores: Dict[str, float],
        attempt: int,
    ) -> bool:
        """
        Return True when faithfulness is below threshold
        and another retry is available.

        Attempt semantics:

            attempt=1, max=2 -> retry
            attempt=2, max=2 -> retry
            attempt=3, max=2 -> no retry

        The decision layer treats max_retry_attempts as the
        maximum retry attempt number. Rejection occurs only
        after that limit has been exceeded.
        """

        faithfulness = scores.get(
            "faithfulness",
            0.0,
        )

        if (
            faithfulness < self.faithfulness_threshold
            and attempt <= self.max_retry_attempts
        ):
            logger.debug(
                "Should retry: "
                f"faithfulness={faithfulness:.2f} < "
                f"{self.faithfulness_threshold} "
                f"(attempt "
                f"{attempt}/{self.max_retry_attempts})"
            )

            return True

        if (
            faithfulness < self.faithfulness_threshold
            and attempt > self.max_retry_attempts
        ):
            logger.warning(
                "Maximum retry attempts exceeded: "
                f"faithfulness={faithfulness:.2f} "
                f"(attempt "
                f"{attempt}/{self.max_retry_attempts})"
            )

        return False

    def should_fallback(
        self,
        scores: Dict[str, float],
    ) -> bool:
        """
        Return True when relevance is below threshold
        and fallback is enabled.
        """

        if not self.fallback_enabled:
            return False

        relevance = scores.get(
            "answer_relevancy",
            0.0,
        )

        if relevance < self.relevance_threshold:
            logger.debug(
                "Should fallback: "
                f"relevance={relevance:.2f} < "
                f"{self.relevance_threshold}"
            )

            return True

        return False

    # ──────────────────────────────────────────────
    # RETRY CONFIGURATION
    # ──────────────────────────────────────────────

    def get_retry_config(
        self,
        attempt: int = 1,
    ) -> RetrievalConfig:
        """
        Build the retrieval configuration for a retry.

        Each retry broadens the retrieval window.

        Example:

            Initial:
                dense = 5
                sparse = 5

            Retry 1:
                dense = 10
                sparse = 10

        Args:
            attempt:
                Current retry attempt number.

        Returns:
            RetrievalConfig
        """

        if attempt < 1:
            raise ValueError(
                "attempt must be >= 1"
            )

        increase = (
            settings.RETRY_RETRIEVAL_TOP_K_INCREASE
            * attempt
        )

        config = RetrievalConfig(
            top_k_dense=(
                settings.TOP_K_DENSE
                + increase
            ),
            top_k_sparse=(
                settings.TOP_K_SPARSE
                + increase
            ),
            score_threshold=(
                settings.RETRY_LOWERED_THRESHOLD
            ),
        )

        logger.info(
            "Retry retrieval config created | "
            f"attempt={attempt} | "
            f"dense_k={config.top_k_dense} | "
            f"sparse_k={config.top_k_sparse} | "
            f"threshold={config.score_threshold}"
        )

        return config

    # ──────────────────────────────────────────────
    # FALLBACK
    # ──────────────────────────────────────────────

    def get_fallback_model(
        self,
    ) -> Optional[str]:
        """
        Return the configured fallback model if it differs
        from the primary model.
        """

        if (
            hasattr(
                self,
                "FALLBACK_LLM_MODEL",
            )
            and self.FALLBACK_LLM_MODEL
        ):
            fallback_model = (
                self.FALLBACK_LLM_MODEL
            )
        else:
            fallback_model = (
                settings.FALLBACK_LLM_MODEL
            )

        main_model = settings.LLM_MODEL

        if (
            fallback_model
            and fallback_model != main_model
        ):
            logger.info(
                f"Falling back from "
                f"{main_model} to "
                f"{fallback_model}"
            )

            return fallback_model

        logger.warning(
            "Fallback model not configured "
            "or same as main model"
        )

        return None

    # ──────────────────────────────────────────────
    # MAIN DECISION
    # ──────────────────────────────────────────────

    def evaluate_and_decide(
        self,
        answer: str,
        scores: Dict[str, float],
        attempt: int = 1,
        max_attempts: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any], bool]:
        """
        Evaluate the current answer and decide what happens next.

        Decision order:

            1. REJECT if retry limit has been exceeded.
            2. RETRY if faithfulness is below threshold and
               another retry is still allowed.
            3. FALLBACK if relevance is below threshold and
               fallback is available.
            4. ACCEPT otherwise.

        Attempt semantics:

            attempt <= max_attempts:
                retry is allowed.

            attempt > max_attempts:
                retry limit has been exceeded and the
                answer is rejected.
        """

        decision_id = str(
            uuid.uuid4()
        )[:8]

        max_attempts = (
            max_attempts
            if max_attempts is not None
            else self.max_retry_attempts
        )

        if attempt < 1:
            raise ValueError(
                "attempt must be >= 1"
            )

        if max_attempts < 1:
            raise ValueError(
                "max_attempts must be >= 1"
            )

        faithfulness = scores.get(
            "faithfulness",
            0.0,
        )

        relevance = scores.get(
            "answer_relevancy",
            0.0,
        )

        decision_type = "UNKNOWN"
        reason = ""
        should_return = False
        metadata_extra: Dict[str, Any] = {}

        # ──────────────────────────────────────────
        # REJECT
        # ──────────────────────────────────────────

        if attempt > max_attempts:

            decision_type = "REJECT"

            reason = (
                f"Maximum attempts "
                f"({max_attempts}) exceeded"
            )

            should_return = True

            logger.warning(
                f"[{decision_id}] "
                "Decision: REJECT"
            )

        # ──────────────────────────────────────────
        # RETRY
        # ──────────────────────────────────────────

        elif (
            faithfulness
            < self.faithfulness_threshold
        ):

            decision_type = "RETRY"

            reason = (
                f"Faithfulness "
                f"{faithfulness:.2f} < "
                f"{self.faithfulness_threshold}"
            )

            retry_config = (
                self.get_retry_config(
                    attempt=attempt
                )
            )

            metadata_extra[
                "retry_config"
            ] = {
                "top_k_dense":
                    retry_config.top_k_dense,
                "top_k_sparse":
                    retry_config.top_k_sparse,
                "score_threshold":
                    retry_config.score_threshold,
            }

            should_return = False

            logger.info(
                f"[{decision_id}] "
                f"Retry requested | "
                f"attempt={attempt}/{max_attempts}"
            )

        # ──────────────────────────────────────────
        # FALLBACK
        # ──────────────────────────────────────────

        elif (
            relevance
            < self.relevance_threshold
        ):

            if self.fallback_enabled:

                fallback_model = (
                    self.get_fallback_model()
                )

                if fallback_model:

                    decision_type = "FALLBACK"

                    reason = (
                        f"Relevance "
                        f"{relevance:.2f} < "
                        f"{self.relevance_threshold}"
                    )

                    metadata_extra[
                        "fallback_model"
                    ] = fallback_model

                    should_return = False

                else:

                    decision_type = "ACCEPT"

                    reason = (
                        f"Relevance "
                        f"{relevance:.2f} < "
                        f"{self.relevance_threshold}, "
                        "but no fallback model "
                        "is available"
                    )

                    should_return = True

            else:

                decision_type = "ACCEPT"

                reason = (
                    f"Relevance "
                    f"{relevance:.2f} < "
                    f"{self.relevance_threshold}, "
                    "but fallback is disabled"
                )

                should_return = True

        # ──────────────────────────────────────────
        # ACCEPT
        # ──────────────────────────────────────────

        else:

            decision_type = "ACCEPT"

            reason = (
                f"Faithfulness "
                f"{faithfulness:.2f} >= "
                f"{self.faithfulness_threshold}, "
                f"Relevance "
                f"{relevance:.2f} >= "
                f"{self.relevance_threshold}"
            )

            should_return = True

        # ──────────────────────────────────────────
        # METADATA
        # ──────────────────────────────────────────

        metadata = {
            "decision_id": decision_id,
            "decision_type": decision_type,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "reason": reason,
            "should_return": should_return,
            "original_answer": answer,
            "scores": {
                "faithfulness": round(
                    faithfulness,
                    3,
                ),
                "answer_relevancy": round(
                    relevance,
                    3,
                ),
            },
            "thresholds": {
                "faithfulness":
                    self.faithfulness_threshold,
                "relevance":
                    self.relevance_threshold,
            },
            "timestamp":
                datetime.now().isoformat(),
            **metadata_extra,
        }

        log_msg = (
            f"[{decision_id}] "
            f"Decision: {decision_type} | "
            f"Attempt {attempt}/{max_attempts} | "
            f"Faithfulness: "
            f"{faithfulness:.2f} | "
            f"Relevance: "
            f"{relevance:.2f}"
        )

        if decision_type == "ACCEPT":

            logger.info(
                f"{log_msg} ACCEPT"
            )

        elif decision_type == "RETRY":

            logger.info(
                f"{log_msg} RETRY"
            )

        elif decision_type == "FALLBACK":

            logger.info(
                f"{log_msg} FALLBACK"
            )

        elif decision_type == "REJECT":

            logger.warning(
                f"{log_msg} REJECT"
            )

        return (
            answer,
            metadata,
            should_return,
        )

    # ──────────────────────────────────────────────
    # UTILITIES
    # ──────────────────────────────────────────────

    def log_decision_history(
        self,
        decisions: List[Dict[str, Any]],
    ) -> None:
        """Log the decision history for one request."""

        if not decisions:
            return

        logger.info(
            f"Decision history: "
            f"{len(decisions)} attempt(s), "
            f"final decision: "
            f"{decisions[-1]['decision_type']}"
        )

        for i, decision in enumerate(
            decisions,
            1,
        ):
            decision_type = decision.get(
                "decision_type",
                "UNKNOWN",
            )

            if (
                "scores" in decision
                and "faithfulness"
                in decision["scores"]
            ):
                faithfulness = decision[
                    "scores"
                ]["faithfulness"]

                logger.debug(
                    f"Attempt {i}: "
                    f"{decision_type} "
                    f"(faith={faithfulness:.2f})"
                )

            else:

                logger.debug(
                    f"Attempt {i}: "
                    f"{decision_type}"
                )

    def get_decision_summary(
        self,
        decision: Dict[str, Any],
    ) -> str:
        """Return a human-readable decision summary."""

        decision_type = decision[
            "decision_type"
        ]

        attempt = decision[
            "attempt"
        ]

        max_attempts = decision[
            "max_attempts"
        ]

        reason = decision[
            "reason"
        ]

        return (
            f"{decision_type} "
            f"(Attempt {attempt}/{max_attempts}): "
            f"{reason}"
        )
