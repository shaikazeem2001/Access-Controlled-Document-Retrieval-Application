"""
rag/prompt.py

Grounded RAG prompt engineering.

Key principles applied here:
  1. CITATION ENFORCEMENT — the model is required to cite [SOURCE:chunk_id]
     for every factual claim. This makes hallucinations detectable:
     a hallucinated claim has no matching source in the context.

  2. EXPLICIT REFUSAL TRIGGER — the model is given a specific phrase to output
     when the answer is not in the context. This phrase is then pattern-matched
     in the guardrails layer to enforce structured refusal.

  3. CONTEXT BEFORE QUESTION — placing the context first, before the question,
     reduces the chance the model "primes" on the question and then fabricates
     supporting evidence.

  4. TEMPERATURE REMINDER — the system prompt reinforces deterministic behaviour.
     LLMs ignore this at inference time if the API temperature is high; always
     set temperature=0.0 at the API level.

  5. NO PREAMBLE INSTRUCTION — prevents the model from opening with
     "Based on the provided context..." which wastes tokens and confuses
     downstream citation parsers.
"""

from langchain_core.prompts import PromptTemplate

# ── System message (injected into ChatOpenAI as system role) ──────────────────

SYSTEM_MESSAGE = """\
You are a precise document analysis assistant. Your ONLY job is to answer \
questions using the Context block below. You have no other knowledge source.

Rules you MUST follow without exception:
1. Every factual claim in your answer MUST cite its source using [SOURCE:chunk_id].
2. If the answer is not contained in the Context, respond with exactly:
   INSUFFICIENT_CONTEXT: <brief explanation of what is missing>
3. Never infer, extrapolate, or guess beyond what is explicitly stated.
4. Never say "based on the context" or similar preamble — just answer directly.
5. Use Markdown for structure (bold, bullets) but keep it concise.
"""

# ── Main RAG prompt template ───────────────────────────────────────────────────

RAG_TEMPLATE = """\
<context>
{context}
</context>

Question: {question}

Answer (cite every claim with [SOURCE:chunk_id], or reply INSUFFICIENT_CONTEXT):"""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_TEMPLATE,
)


# ── Context builder ────────────────────────────────────────────────────────────

def build_context_block(docs) -> str:
    """
    Serialise retrieved documents into the <context> block.

    Each chunk is labelled with its chunk_id so the model can cite it,
    and its source filename and page so the user can verify it.

    Format:
        [chunk_id=abc123 | source=contract.pdf | page=3]
        <text of the chunk>
        ---
    """
    parts = []
    for doc in docs:
        meta     = doc.metadata
        chunk_id = meta.get("chunk_id", "unknown")
        source   = meta.get("source", "unknown")
        page     = meta.get("page", "?")
        text     = doc.page_content.strip()

        parts.append(
            f"[chunk_id={chunk_id} | source={source} | page={page}]\n{text}"
        )

    return "\n---\n".join(parts)


def format_prompt(question: str, docs) -> str:
    """Return the fully formatted prompt string (for logging/debugging)."""
    context = build_context_block(docs)
    return RAG_PROMPT.format(context=context, question=question)