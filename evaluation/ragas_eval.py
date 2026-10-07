# evaluation/ragas_eval.py

import json
import math
import os
import re
import time
from typing import Any, Dict, List

from observability.logger import get_logger
from configs import settings

logger = get_logger("evaluation.ragas")

from embeddings.embedding_model import get_embedding_model


# ─────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────

def _get_embeddings():
    return get_embedding_model()

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _safe(val, default=0.0):
    try:
        return default if (val is None or math.isnan(val) or math.isinf(val)) else float(val)
    except Exception:
        return default


def _clean_answer(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\[(?:SOURCE|src):.*?\]", "", text, flags=re.IGNORECASE)
    return text.strip()


def _trim_contexts(contexts: List[str]) -> List[str]:
    max_contexts = getattr(settings, "RAGAS_MAX_CONTEXTS", 10)
    max_chars = getattr(settings, "RAGAS_CONTEXT_MAX_CHARS", 4500)

    contexts = list(dict.fromkeys(contexts))
    dropped_count = max(0, len(contexts) - max_contexts)
    selected = contexts[:max_contexts]

    truncated_count = sum(1 for ctx in selected if len(ctx) > max_chars)
    if truncated_count or dropped_count:
        logger.warning(
            f"_trim_contexts: dropped {dropped_count} context(s) beyond "
            f"RAGAS_MAX_CONTEXTS={max_contexts}, truncated {truncated_count} "
            f"context(s) beyond RAGAS_CONTEXT_MAX_CHARS={max_chars}. "
            f"RAGAS faithfulness will only see the retained portion — "
            f"a citation pointing past the cutoff can score as unfaithful "
            f"even though the full chunk supports it."
        )

    return [ctx[:max_chars] for ctx in selected]


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if error is a rate limit error from Groq API."""
    error_str = str(error).lower()
    return "429" in error_str or "rate_limit" in error_str or "rate limit" in error_str


def _is_json_parse_error(error: Exception) -> bool:
    """
    Check if error is a JSON parsing/output format error.
    
    Common patterns:
    - JSONDecodeError
    - OUTPUT_PARSING_FAILURE
    - "json" or "parse" in error message
    - Llama reasoning before JSON in response
    """
    error_str = str(error).lower()
    
    # Direct error type checks
    if "json" in error_str or "parse" in error_str or "parsing" in error_str:
        return True
    if "output_parsing" in error_str or "output parsing" in error_str:
        return True
    if "expected json" in error_str or "expected valid json" in error_str:
        return True
    
    # Check error class name
    error_type = type(error).__name__.lower()
    if "json" in error_type or "parse" in error_type:
        return True
    
    return False


def _exponential_backoff_wait(attempt: int, base_wait: int = 5) -> int:
    """
    Calculate exponential backoff wait time.
    
    Args:
        attempt: Current attempt number (0-indexed)
        base_wait: Base wait time in seconds (default 5)
    
    Returns:
        Time to wait in seconds (base_wait * 2^attempt)
        
    Examples:
        attempt=0: wait = 5 * 2^0 = 5 seconds
        attempt=1: wait = 5 * 2^1 = 10 seconds
        attempt=2: wait = 5 * 2^2 = 20 seconds
    """
    return base_wait * (2 ** attempt)


# ─────────────────────────────────────────────────────────────
# LLM SETUP (FINAL - GROQ ONLY)
# ─────────────────────────────────────────────────────────────
def _get_llm():
    from langchain_openai import ChatOpenAI

    api_key = settings.LLM_API_KEY
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    class GroqLLM(ChatOpenAI):
        """
        Groq LLM wrapper with rate limit resilience.
        Forces n=1 to avoid Groq API constraint issues.
        """
        
        def _force_n(self, kwargs):
            kwargs["n"] = 1
            return kwargs

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            kwargs = self._force_n(kwargs)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        def generate(self, messages, stop=None, **kwargs):
            kwargs = self._force_n(kwargs)
            return super().generate(messages, stop=stop, **kwargs)

        async def agenerate(self, messages, stop=None, **kwargs):
            kwargs = self._force_n(kwargs)
            return await super().agenerate(messages, stop=stop, **kwargs)

        def invoke(self, input, config=None, **kwargs):
            kwargs = self._force_n(kwargs)
            return super().invoke(input, config=config, **kwargs)

        async def ainvoke(self, input, config=None, **kwargs):
            kwargs = self._force_n(kwargs)
            return await super().ainvoke(input, config=config, **kwargs)

    return GroqLLM(
        api_key=api_key,
        base_url=settings.LLM_API_BASE,
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        # RAGAS's internal prompts (claim decomposition, NLI verification) require
        # strict JSON-only output. Llama-via-Groq tends to add reasoning text before
        # the JSON block, which breaks LangChain's JSON output parser
        # (OUTPUT_PARSING_FAILURE). Forcing JSON mode here is safe — this LLM
        # instance is only ever used internally by RAGAS, never to generate
        # user-facing answers.
        model_kwargs={"response_format": {"type": "json_object"}},
    )


# ─────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────

def run_ragas_evaluation(test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    import ragas
    logger.info(f"RAGAS version: {ragas.__version__}")

    # prevent accidental OpenAI usage
    os.environ.pop("OPENAI_API_KEY", None)

    # ── Clean + Filter test cases ─────────────────────────────
    cleaned_cases = []
    for tc in test_cases:
        if not tc.get("answer") or not tc.get("contexts"):
            continue

        answer = _clean_answer(tc["answer"])
        contexts = _trim_contexts(tc["contexts"])

        if not answer or not contexts:
            continue

        cleaned_cases.append({
            "question": tc["question"],
            "answer": answer,
            "contexts": contexts,
        })

    if not cleaned_cases:
        return {
            "error": "No valid cleaned test cases",
            "total_cases": len(test_cases),
            "evaluated_cases": 0,
        }

    logger.info(f"Evaluating {len(cleaned_cases)} cleaned cases")

    # ── Initialize LLM + embeddings ───────────────────────────
    llm = _get_llm()
    embeds = _get_embeddings()

    if llm is None:
        raise ValueError("LLM initialization failed (None)")

    # ── RAGAS Setup ───────────────────────────────────────────
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.run_config import RunConfig
    from ragas.metrics import Faithfulness, AnswerRelevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    ragas_llm = LangchainLLMWrapper(langchain_llm=llm)
    ragas_embed = LangchainEmbeddingsWrapper(embeddings=embeds)

    faithfulness = Faithfulness(llm=ragas_llm)
    answer_relevancy = AnswerRelevancy(
        llm=ragas_llm,
        embeddings=ragas_embed
    )

    # ── Build dataset ─────────────────────────────────────────
    samples = [
        SingleTurnSample(
            user_input=tc["question"],
            response=tc["answer"],
            retrieved_contexts=tc["contexts"],
        )
        for tc in cleaned_cases
    ]

    import nest_asyncio
    nest_asyncio.apply()

    # ── ENHANCED: Retry with exponential backoff for ALL error types ───────────
    retry_count = getattr(settings, "RAGAS_RETRY_COUNT", 2)
    max_rate_limit_retries = 3  # Extra retries for rate limits
    rate_limit_attempt = 0
    json_parse_attempt = 0
    last_exception = None
    df = None

    for attempt in range(retry_count):
        try:
            result = evaluate(
                dataset=EvaluationDataset(samples=samples),
                metrics=[faithfulness, answer_relevancy],
                run_config=RunConfig(max_workers=1, timeout=180),
            )
            df = result.to_pandas()
            logger.info(f"RAGAS evaluation succeeded on attempt {attempt + 1}")
            break
            
        except Exception as e:
            is_rate_limit = _is_rate_limit_error(e)
            is_json_error = _is_json_parse_error(e)
            error_type = type(e).__name__
            
            # ── Rate limit error: exponential backoff ──────────────────────────
            if is_rate_limit and rate_limit_attempt < max_rate_limit_retries:
                wait_time = _exponential_backoff_wait(rate_limit_attempt)
                logger.warning(
                    f"Rate limit hit ({error_type}). "
                    f"Backoff attempt {rate_limit_attempt + 1}/{max_rate_limit_retries}. "
                    f"Waiting {wait_time}s before retry..."
                )
                rate_limit_attempt += 1
                time.sleep(wait_time)
                continue  # Retry without incrementing main attempt counter
            
            # ── JSON parse error: exponential backoff ──────────────────────────
            # These are common with Llama reasoning before JSON. Backoff helps
            # as Groq's server state may recover. After backoff fails, we still
            # process partial results (per-metric fallback below).
            elif is_json_error:
                if json_parse_attempt < 2:  # Allow 2 backoff attempts for JSON errors
                    wait_time = _exponential_backoff_wait(json_parse_attempt)
                    logger.warning(
                        f"JSON parse error ({error_type}). "
                        f"This often happens when Llama adds reasoning before JSON. "
                        f"Backoff attempt {json_parse_attempt + 1}/2. "
                        f"Waiting {wait_time}s before retry..."
                    )
                    json_parse_attempt += 1
                    time.sleep(wait_time)
                    continue  # Retry without incrementing main attempt counter
                else:
                    # JSON backoff exhausted, log and fall through to fallback logic
                    logger.error(
                        f"JSON parse error ({error_type}) persists after backoff. "
                        f"Will attempt per-metric partial evaluation if available."
                    )
                    last_exception = e
                    break  # Exit loop to try partial results if they exist
            
            # ── Other errors: regular retry ────────────────────────────────────
            else:
                logger.warning(
                    f"RAGAS attempt {attempt + 1}/{retry_count} failed ({error_type}): {e}"
                )
                last_exception = e
                time.sleep(2)  # Brief pause before next attempt
    else:
        # Loop completed without success
        if df is None:
            return {
                "error": f"Evaluation failed after {retry_count} attempts. "
                         f"Last error ({type(last_exception).__name__ if last_exception else 'Unknown'}): "
                         f"{last_exception}",
                "evaluated_cases": 0,
            }

    # ── Process results ───────────────────────────────────────
    # IMPORTANT: faithfulness and answer_relevancy can fail independently
    # (e.g. Faithfulness hits a JSON parse error while Answer Relevancy
    # succeeds for the same row). Score each metric on its own valid values
    # rather than discarding a row entirely because one metric is NaN —
    # otherwise a single bad parse on one metric silently throws away a
    # perfectly good score from the other.
    
    if df is None:
        return {
            "error": f"No evaluation results available. Last error: {last_exception}",
            "evaluated_cases": 0,
        }
    
    faith_vals: List[float] = []
    relev_vals: List[float] = []
    fully_failed_count = 0  # rows where BOTH metrics failed

    for i in range(len(df)):
        f_val = df.iloc[i].get("faithfulness")
        r_val = df.iloc[i].get("answer_relevancy")

        f_ok = f_val is not None and not math.isnan(f_val)
        r_ok = r_val is not None and not math.isnan(r_val)

        if f_ok:
            faith_vals.append(f_val)
        if r_ok:
            relev_vals.append(r_val)
        if not f_ok and not r_ok:
            fully_failed_count += 1
        if not f_ok:
            logger.warning(
                f"Faithfulness failed for case {i} (likely judge-LLM parse error) "
                f"— excluded from average"
            )
        if not r_ok:
            logger.warning(
                f"Answer relevancy failed for case {i} "
                f"— excluded from average"
            )

    if not faith_vals and not relev_vals:
        return {
            "error": "All evaluations failed for every metric",
            "evaluated_cases": 0,
            "failed_cases": fully_failed_count,
        }

    faith_avg = (sum(faith_vals) / len(faith_vals)) if faith_vals else None
    relev_avg = (sum(relev_vals) / len(relev_vals)) if relev_vals else None

    # ── Per question breakdown ────────────────────────────────
    per_question = []

    for i in range(len(df)):
        f_val = df.iloc[i].get("faithfulness")
        r_val = df.iloc[i].get("answer_relevancy")

        f_ok = f_val is not None and not math.isnan(f_val)
        r_ok = r_val is not None and not math.isnan(r_val)

        if not f_ok and not r_ok:
            continue  # nothing usable for this case at all

        per_question.append({
            "question": cleaned_cases[i]["question"],
            "faithfulness": round(_safe(f_val), 3) if f_ok else None,
            "answer_relevancy": round(_safe(r_val), 3) if r_ok else None,
        })

    # ── Final output ──────────────────────────────────────────
    success_count = len(per_question)
    logger.info(
        f"RAGAS evaluation complete: {success_count} successful, "
        f"{fully_failed_count} fully failed out of {len(cleaned_cases)} cases"
    )
    
    return {
        "evaluated_cases": len(per_question),
        "failed_cases": fully_failed_count,
        "total_cases": len(test_cases),
        "faithfulness": round(faith_avg, 3) if faith_avg is not None else None,
        "answer_relevancy": round(relev_avg, 3) if relev_avg is not None else None,
        "hallucination_rate": round(1.0 - faith_avg, 3) if faith_avg is not None else None,
        "per_question": per_question,
    }