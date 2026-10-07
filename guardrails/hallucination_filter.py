"""
guardrails/hallucination_filter.py

Post-generation hallucination detection and refusal enforcement.

Grounding is checked at the CLAIM level rather than only checking
whether citation IDs exist.

Checks:

1. INSUFFICIENT_CONTEXT
   Detect when the model explicitly says it cannot answer.

2. CITATION VALIDATION
   Verify that [SOURCE:chunk_id] references actually exist in the
   retrieved documents.

3. CLAIM-LEVEL GROUNDING
   Extract individual claims, associate them with their cited sources,
   and use an NLI model to determine whether the cited context actually
   entails each claim.

Grounding score:

    supported_claims / total_claims

A valid citation alone does NOT make a claim grounded.

The cited context must actually entail the generated claim.

NLI model:

    cross-encoder/nli-deberta-v3-base

The NLI model produces scores for:

    CONTRADICTION
    ENTAILMENT
    NEUTRAL

A claim is considered supported when its best entailment probability
reaches GROUNDING_NLI_THRESHOLD.
"""

import math
import re
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from configs.settings import settings
from observability.logger import get_logger


logger = get_logger("guardrails.hallucination_filter")


# ── Patterns ──────────────────────────────────────────────────────────────────

_INSUFFICIENT_PATTERN = re.compile(
    r"INSUFFICIENT_CONTEXT\s*:",
    re.IGNORECASE,
)

_CITATION_PATTERN = re.compile(
    r"\[SOURCE:([a-zA-Z0-9_\-]+)\]",
)


# ── NLI model ──────────────────────────────────────────────────────────────────

_nli_model = None


def _get_nli_model() -> CrossEncoder:
    """
    Load the NLI model lazily.

    The model is loaded only when claim-level grounding is performed.
    This avoids loading the model during normal module imports.
    """

    global _nli_model

    if _nli_model is None:
        model_name = getattr(
            settings,
            "GROUNDING_NLI_MODEL",
            "cross-encoder/nli-deberta-v3-base",
        )

        logger.info(
            f"Loading grounding NLI model: {model_name}"
        )

        _nli_model = CrossEncoder(model_name)

        logger.info(
            "Grounding NLI model loaded successfully"
        )

    return _nli_model


# ── NLI helpers ───────────────────────────────────────────────────────────────

def _get_entailment_index(model: CrossEncoder) -> int:
    """
    Determine which output index represents ENTAILMENT.

    nli-deberta-v3-base normally exposes:

        0 -> contradiction
        1 -> entailment
        2 -> neutral

    We inspect the model configuration instead of blindly depending
    on a hard-coded ordering.

    Falls back to index 1 because that is the standard ordering for
    the configured model.
    """

    try:
        id2label = model.model.config.id2label

        if id2label:
            for index, label in id2label.items():
                normalized = str(label).strip().lower()

                if normalized in {
                    "entailment",
                    "entails",
                    "label_entailment",
                }:
                    return int(index)

    except Exception as exc:
        logger.warning(
            f"Could not determine NLI label mapping: {exc}"
        )

    return 1


def _prediction_to_probabilities(
    prediction,
) -> List[float]:
    """
    Convert an NLI prediction into probabilities.

    CrossEncoder may return:

    - 3 logits
    - 3 probabilities
    - a scalar

    For a three-class NLI model, logits are converted with softmax.

    If the values already look like probabilities, they are retained.
    """

    try:
        values = [float(value) for value in prediction]
    except TypeError:
        return [float(prediction)]

    if not values:
        return []

    # Already looks like probabilities.
    if (
        all(0.0 <= value <= 1.0 for value in values)
        and abs(sum(values) - 1.0) < 1e-3
    ):
        return values

    # Convert logits to probabilities.
    max_value = max(values)

    exp_values = [
        math.exp(value - max_value)
        for value in values
    ]

    total = sum(exp_values)

    if total <= 0:
        return [0.0 for _ in values]

    return [
        value / total
        for value in exp_values
    ]


# ── Claim extraction ──────────────────────────────────────────────────────────

def _extract_claims(answer: str) -> List[str]:
    """
    Extract sentence-level claims from the generated answer.

    Citations are removed before returning the claims.

    This function intentionally remains deterministic. It does not
    make another LLM call for claim decomposition.

    Returns:
        List[str]
    """

    if not answer:
        return []

    cleaned = _CITATION_PATTERN.sub(
        "",
        answer,
    )

    # Normalize whitespace.
    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    ).strip()

    if not cleaned:
        return []

    # Split on sentence boundaries.
    raw_claims = re.split(
        r"(?<=[.!?])\s+",
        cleaned,
    )

    claims = []

    for claim in raw_claims:
        claim = claim.strip()

        # Remove common markdown list markers.
        claim = re.sub(
            r"^(?:[-*•]|\d+[.)])\s+",
            "",
            claim,
        ).strip()

        if not claim:
            continue

        # Ignore extremely short fragments.
        if len(claim) < 10:
            continue

        claims.append(claim)

    return claims


def _extract_claims_with_citations(
    answer: str,
) -> List[Tuple[str, List[str]]]:
    """
    Extract claims while preserving the citation relationship.

    Example:

        Falcon reduced costs by 20% [SOURCE:c1].
        Revenue increased by 15% [SOURCE:c2].

    becomes:

        [
            ("Falcon reduced costs by 20%.", ["c1"]),
            ("Revenue increased by 15%.", ["c2"]),
        ]

    Claims without citations are returned with an empty citation list.
    """

    if not answer:
        return []

    # Keep citations during sentence splitting.
    cleaned = re.sub(
        r"\s+",
        " ",
        answer,
    ).strip()

    if not cleaned:
        return []

    raw_claims = re.split(
        r"(?<=[.!?])\s+",
        cleaned,
    )

    claims_with_citations = []

    for raw_claim in raw_claims:
        raw_claim = raw_claim.strip()

        if not raw_claim:
            continue

        citations = _CITATION_PATTERN.findall(
            raw_claim
        )

        claim = _CITATION_PATTERN.sub(
            "",
            raw_claim,
        )

        # Remove markdown list markers.
        claim = re.sub(
            r"^(?:[-*•]|\d+[.)])\s+",
            "",
            claim,
        ).strip()

        if not claim:
            continue

        if len(claim) < 10:
            continue

        claims_with_citations.append(
            (
                claim,
                list(dict.fromkeys(citations)),
            )
        )

    return claims_with_citations


# ── Claim-level NLI ───────────────────────────────────────────────────────────

def _check_claim_support(
    claim: str,
    contexts: List[str],
) -> Tuple[bool, float]:
    """
    Determine whether at least one context entails the claim.

    Args:
        claim:
            Generated factual claim.

        contexts:
            Candidate supporting contexts.

    Returns:
        (supported, best_entailment_score)

    NLI formulation:

        premise   = retrieved context
        hypothesis = generated claim

    The claim is supported when the strongest entailment probability
    reaches GROUNDING_NLI_THRESHOLD.
    """

    if not claim or not contexts:
        return False, 0.0

    model = _get_nli_model()

    pairs = [
        [context, claim]
        for context in contexts
        if context and context.strip()
    ]

    if not pairs:
        return False, 0.0

    predictions = model.predict(
        pairs
    )

    if not isinstance(predictions, (list, tuple)):
        predictions = [predictions]

    entailment_index = _get_entailment_index(
        model
    )

    best_entailment = 0.0

    for prediction in predictions:
        probabilities = _prediction_to_probabilities(
            prediction
        )

        if not probabilities:
            continue

        if entailment_index >= len(probabilities):
            logger.warning(
                "NLI entailment index is outside "
                "the model output range"
            )
            continue

        entailment_score = float(
            probabilities[entailment_index]
        )

        best_entailment = max(
            best_entailment,
            entailment_score,
        )

    threshold = float(
        getattr(
            settings,
            "GROUNDING_NLI_THRESHOLD",
            0.70,
        )
    )

    supported = (
        best_entailment >= threshold
    )

    return (
        supported,
        best_entailment,
    )


# ── Citation-specific claim grounding ─────────────────────────────────────────

def _check_claim_against_citations(
    claim: str,
    cited_ids: List[str],
    documents_by_id: Dict[str, Document],
) -> Tuple[bool, float]:
    """
    Check a claim specifically against its cited documents.

    This is the key distinction from the old grounding implementation.

    A citation is not considered valid evidence merely because the
    chunk exists.

    The cited chunk must actually entail the claim.
    """

    if not claim or not cited_ids:
        return False, 0.0

    contexts = []

    for chunk_id in cited_ids:
        document = documents_by_id.get(
            chunk_id
        )

        if document is None:
            continue

        if not document.page_content:
            continue

        content = document.page_content.strip()

        if content:
            contexts.append(content)

    if not contexts:
        return False, 0.0

    return _check_claim_support(
        claim,
        contexts,
    )


# ── Main guardrail ────────────────────────────────────────────────────────────

def check_response(
    answer: str,
    retrieved_docs: List[Document],
) -> Tuple[bool, float, str]:
    """
    Validate a generated answer against retrieved context.

    Returns:

        (
            is_grounded,
            grounding_score,
            final_answer,
        )

    grounding_score:

        supported_claims / total_claims

    A claim is supported only when:

        1. It contains a valid citation.
        2. The cited chunk exists in retrieved_docs.
        3. The cited context entails the claim according to NLI.

    Therefore:

        valid citation != grounded claim
    """

    # ── Check 1: explicit insufficient context ───────────────────────────────

    if _INSUFFICIENT_PATTERN.search(
        answer or ""
    ):
        logger.info(
            "Model reported INSUFFICIENT_CONTEXT"
        )

        return (
            False,
            0.0,
            settings.REFUSAL_MESSAGE,
        )

    if not answer or not answer.strip():
        logger.warning(
            "Empty answer received"
        )

        return (
            False,
            0.0,
            settings.REFUSAL_MESSAGE,
        )

    # ── Build valid citation lookup ──────────────────────────────────────────

    documents_by_id: Dict[str, Document] = {}

    for document in retrieved_docs:
        chunk_id = document.metadata.get(
            "chunk_id"
        )

        if chunk_id:
            documents_by_id[str(chunk_id)] = document

    valid_chunk_ids = set(
        documents_by_id.keys()
    )

    # ── Check 2: citation validation ─────────────────────────────────────────

    cited_ids = set(
        _CITATION_PATTERN.findall(
            answer
        )
    )

    if not cited_ids:
        logger.warning(
            "Answer contains no citations"
        )

        return (
            False,
            0.0,
            settings.REFUSAL_MESSAGE,
        )

    hallucinated_citations = (
        cited_ids - valid_chunk_ids
    )

    if hallucinated_citations:
        logger.warning(
            "Hallucinated citation IDs detected: "
            f"{hallucinated_citations}"
        )

        return (
            False,
            0.0,
            settings.REFUSAL_MESSAGE,
        )

    logger.debug(
        f"Validated {len(cited_ids)} citation IDs"
    )

    # ── Check 3: claim extraction ────────────────────────────────────────────

    claims_with_citations = (
        _extract_claims_with_citations(
            answer
        )
    )

    if not claims_with_citations:
        logger.warning(
            "No factual claims could be extracted "
            "from answer"
        )

        return (
            False,
            0.0,
            settings.REFUSAL_MESSAGE,
        )

    # ── Check each claim ─────────────────────────────────────────────────────

    supported_claims = 0

    claim_results = []

    for claim, claim_citations in claims_with_citations:

        # Every factual claim must identify its evidence.
        if not claim_citations:
            logger.warning(
                "Claim has no citation: "
                f"{claim[:120]}"
            )

            claim_results.append(
                {
                    "claim": claim,
                    "citations": [],
                    "supported": False,
                    "score": 0.0,
                }
            )

            continue

        # Verify every citation attached to this claim.
        invalid_claim_citations = [
            citation
            for citation in claim_citations
            if citation not in valid_chunk_ids
        ]

        if invalid_claim_citations:
            logger.warning(
                "Claim contains invalid citations: "
                f"{invalid_claim_citations}"
            )

            claim_results.append(
                {
                    "claim": claim,
                    "citations": claim_citations,
                    "supported": False,
                    "score": 0.0,
                }
            )

            continue

        # NLI against the actual cited evidence.
        supported, score = (
            _check_claim_against_citations(
                claim,
                claim_citations,
                documents_by_id,
            )
        )

        if supported:
            supported_claims += 1

        claim_results.append(
            {
                "claim": claim,
                "citations": claim_citations,
                "supported": supported,
                "score": round(score, 3),
            }
        )

        logger.debug(
            "Claim grounding | "
            f"supported={supported} | "
            f"score={score:.3f} | "
            f"citations={claim_citations} | "
            f"claim={claim[:120]}"
        )

    # ── Calculate grounding score ────────────────────────────────────────────

    total_claims = len(
        claims_with_citations
    )

    grounding_score = (
        supported_claims / total_claims
        if total_claims
        else 0.0
    )

    logger.info(
        "Claim-level grounding: "
        f"{supported_claims}/{total_claims} "
        f"claims supported "
        f"(score={grounding_score:.2f})"
    )

    # ── Apply overall grounding threshold ────────────────────────────────────

    threshold = float(
        getattr(
            settings,
            "GROUNDING_MIN_SCORE",
            0.5,
        )
    )

    if grounding_score < threshold:
        logger.warning(
            "Claim-level grounding score "
            f"{grounding_score:.2f} < threshold "
            f"{threshold:.2f}. Refusing response."
        )

        return (
            False,
            grounding_score,
            settings.REFUSAL_MESSAGE,
        )

    logger.info(
        "Response passed claim-level grounding"
    )

    return (
        True,
        grounding_score,
        answer,
    )