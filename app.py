"""Milestone 5 grounded generation app for the NOVA RAG project.

This module connects the local retrieval engine to the Groq LLM API and wraps
it in a Gradio interface. It is designed to fail gracefully when the API key is
missing or when the network/API call fails.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import gradio as gr
from dotenv import load_dotenv
from groq import Groq

from retrieval_engine import get_chroma_collection, get_model, retrieve_chunks


MODEL_NAME = "llama-3.3-70b-versatile"
TOP_K = 5
RETRIEVAL_CANDIDATE_POOL = 50
FALLBACK_MESSAGE = "I don't have enough information on that."

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "do",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "use",
    "what",
    "when",
    "which",
    "with",
}


# Load local environment variables first so GROQ_API_KEY is available from .env.
load_dotenv()


def _safe_text(value: object) -> str:
    return "" if value is None else str(value)


def _normalize_terms(text: str) -> List[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def _label_for_index(index: int) -> str:
    """Convert 0 -> Source A, 1 -> Source B, ... 25 -> Source Z, 26 -> Source AA."""

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    number = index
    label = ""
    while True:
        number, remainder = divmod(number, 26)
        label = alphabet[remainder] + label
        if number == 0:
            break
        number -= 1
    return f"Source {label}"


@lru_cache(maxsize=1)
def get_retrieval_resources():
    """Load the persistent ChromaDB collection and embedding model once."""

    collection = get_chroma_collection()
    model = get_model()
    return collection, model


@lru_cache(maxsize=1)
def get_groq_client() -> Tuple[Optional[Groq], Optional[str]]:
    """Create the Groq client if the API key is configured.

    Returns
    -------
    (client, error_message)
        Exactly one of the tuple items will be usable.
    """

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None, "GROQ_API_KEY is not configured. Add it to your .env file to enable answering."

    try:
        return Groq(api_key=api_key), None
    except Exception as exc:  # pragma: no cover - defensive guard around client init
        return None, f"Unable to initialize Groq client: {exc}"


def _dedupe_sources(retrieved_chunks: Sequence[Dict[str, object]]) -> Tuple[List[str], Dict[Tuple[str, str], str]]:
    """Build the deduplicated source list and label map used by the prompt."""

    sources: List[str] = []
    label_map: Dict[Tuple[str, str], str] = {}

    for item in retrieved_chunks:
        metadata = cast(Dict[str, Any], item.get("metadata") or {})
        source_url = _safe_text(cast(Any, item.get("source_url") or metadata.get("source_url") or metadata.get("url")))
        file_reference = _safe_text(cast(Any, item.get("file_reference") or metadata.get("source_file") or metadata.get("file_reference")))
        source_key = (source_url, file_reference)

        if source_key in label_map:
            continue

        label = _label_for_index(len(label_map))
        label_map[source_key] = label
        sources.append(f"{label}: {source_url or 'unknown URL'} / {file_reference or 'unknown file'}")

    return sources, label_map


def _build_context(retrieved_chunks: Sequence[Dict[str, object]]) -> Tuple[str, List[str]]:
    """Format retrieved chunks into labeled blocks for the LLM context."""

    sources, label_map = _dedupe_sources(retrieved_chunks)
    blocks: List[str] = []

    for item in retrieved_chunks:
        metadata = cast(Dict[str, Any], item.get("metadata") or {})
        source_url = _safe_text(cast(Any, item.get("source_url") or metadata.get("source_url") or metadata.get("url")))
        file_reference = _safe_text(cast(Any, item.get("file_reference") or metadata.get("source_file") or metadata.get("file_reference")))
        source_key = (source_url, file_reference)
        label = label_map[source_key]
        text = _safe_text(item.get("text", "")).strip()
        distance = item.get("distance")
        similarity = item.get("similarity")

        blocks.append(
            "\n".join(
                [
                    f"[{label}]",
                    f"File: {file_reference or 'unknown file'}",
                    f"URL: {source_url or 'unknown URL'}",
                    f"Distance: {distance}",
                    f"Similarity: {similarity}",
                    "Text:",
                    text,
                ]
            ).strip()
        )

    return "\n\n".join(blocks).strip(), sources


def _candidate_text(item: Dict[str, object]) -> str:
    metadata = cast(Dict[str, Any], item.get("metadata") or {})
    parts = [
        _safe_text(item.get("text", "")),
        _safe_text(item.get("source_url", "")),
        _safe_text(item.get("file_reference", "")),
        _safe_text(metadata.get("post_title", "")),
        _safe_text(metadata.get("source_file", "")),
        _safe_text(metadata.get("source_url", "")),
    ]
    return " ".join(part for part in parts if part).lower()


def _chunk_key(item: Dict[str, object]) -> Tuple[str, str, str]:
    """Stable identity for a chunk, used to drop near-duplicate retrievals."""

    metadata = cast(Dict[str, Any], item.get("metadata") or {})
    source_url = _safe_text(item.get("source_url") or metadata.get("source_url") or metadata.get("url"))
    file_reference = _safe_text(item.get("file_reference") or metadata.get("source_file") or metadata.get("file_reference"))
    comment_id = _safe_text(metadata.get("comment_id") or metadata.get("chunk_id"))
    return (source_url, file_reference, comment_id)


def _dedupe_in_order(items: Sequence[Dict[str, object]], limit: int) -> List[Dict[str, object]]:
    """Keep the first occurrence of each chunk key, preserving order, up to `limit`."""

    selected: List[Dict[str, object]] = []
    seen_keys: set[Tuple[str, str, str]] = set()
    for item in items:
        key = _chunk_key(item)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def _rerank_retrieved_chunks(question: str, retrieved_chunks: Sequence[Dict[str, object]], limit: int) -> List[Dict[str, object]]:
    """Promote chunks that match the question literally while keeping semantic recall."""

    terms = _normalize_terms(question)
    if not terms:
        return _dedupe_in_order(retrieved_chunks, limit)

    # Only apply the programming-language boost when the *question* is actually about
    # languages. Gating on the question (not just the chunk) prevents unrelated
    # questions from promoting chunks that merely happen to mention Java/Python.
    question_about_language = bool({"language", "languages", "python", "java", "c"} & set(terms))

    scored_chunks: List[Tuple[float, float, float, Dict[str, object]]] = []
    for item in retrieved_chunks:
        text = _candidate_text(item)
        semantic_score = float(item.get("similarity") or 0.0)
        lexical_hits = sum(1 for term in terms if term in text)
        # Normalize lexical overlap into [0, 1] so a chunk that merely repeats many
        # query tokens cannot overpower a chunk that is semantically more relevant.
        # The boost is a bounded tie-breaker, not a competing signal.
        lexical_fraction = lexical_hits / len(terms)
        language_bonus = 0.0

        if question_about_language and ("python" in text or "java" in text):
            mentions_course = any(code in text for code in ("csc 221", "csc221", "csc 222", "csc 223"))
            language_bonus = 0.75 if mentions_course else 0.5

        combined = semantic_score + (0.1 * lexical_fraction) + language_bonus
        scored_chunks.append((combined, lexical_hits, semantic_score, item))

    scored_chunks.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)

    return _dedupe_in_order([item for _, _, _, item in scored_chunks], limit)


def _build_system_prompt() -> str:
    return (
        "You are the Unofficial NOVA Student Guide assistant. "
        "Answer the user's question only using the provided context blocks. "
        "Do not use outside knowledge, training data, assumptions, or guesses. "
        "Cite every factual claim inline using the source labels exactly as written, such as [Source A]. "
        "If the provided context does not contain enough specific facts to answer the question, respond exactly with: "
        f"{FALLBACK_MESSAGE}"
    )


def ask(question: str) -> Dict[str, object]:
    """Answer a question using retrieved chunks and the Groq LLM.

    Returns a dictionary with the generated answer and the deduplicated source list.
    The function never raises to the caller; all recoverable failures are converted
    into a user-facing answer string.
    """

    question = _safe_text(question).strip()
    if not question:
        return {"answer": "Please enter a question.", "sources": []}

    try:
        collection, model = get_retrieval_resources()
    except Exception as exc:
        return {"answer": f"Retrieval resources are unavailable: {exc}", "sources": []}

    try:
        retrieved_chunks = retrieve_chunks(question, top_k=RETRIEVAL_CANDIDATE_POOL, collection=collection, model=model)
    except Exception as exc:
        return {"answer": f"Retrieval failed: {exc}", "sources": []}

    retrieved_chunks = _rerank_retrieved_chunks(question, retrieved_chunks, TOP_K)

    context, sources = _build_context(retrieved_chunks)
    if not context.strip():
        return {"answer": FALLBACK_MESSAGE, "sources": []}

    client, client_error = get_groq_client()
    if client_error:
        return {"answer": client_error, "sources": sources}
    if client is None:
        return {"answer": "Groq client is unavailable.", "sources": sources}

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _build_system_prompt()},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Context:\n{context}\n\n"
                "Follow the grounding rules exactly. Use inline citations like [Source A]."
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=cast(Any, messages),
            temperature=0.1,
            top_p=1.0,
            max_tokens=700,
        )
        answer = _safe_text(response.choices[0].message.content).strip()
    except Exception as exc:
        return {"answer": f"Groq request failed: {exc}", "sources": sources}

    if not answer:
        answer = FALLBACK_MESSAGE

    return {"answer": answer, "sources": sources}


def _ask_for_gradio(question: str) -> Tuple[str, str]:
    """Thin wrapper that adapts `ask()` to Gradio's tuple-based outputs."""

    result = ask(question)
    answer = _safe_text(result.get("answer", ""))
    sources = cast(List[str], result.get("sources", []))
    sources_text = "\n".join(_safe_text(source) for source in sources) if sources else ""
    return answer, sources_text


def build_demo() -> gr.Blocks:
    """Construct the Gradio web interface."""

    with gr.Blocks(title="NOVA Unofficial Student Guide") as demo:
        gr.Markdown("# NOVA Unofficial Student Guide\nAsk a question and the assistant will answer only from retrieved NOVA student sources.")
        with gr.Column():
            question_box = gr.Textbox(
                label="Your question",
                placeholder="Ask about NOVA CS, transfer paths, physics, or course advice...",
                lines=2,
            )
            ask_button = gr.Button("Ask", variant="primary")
            answer_box = gr.Textbox(label="Answer", lines=10, interactive=False)
            sources_box = gr.Textbox(label="Retrieved Sources", lines=6, interactive=False)

        question_box.submit(_ask_for_gradio, inputs=question_box, outputs=[answer_box, sources_box])
        ask_button.click(_ask_for_gradio, inputs=question_box, outputs=[answer_box, sources_box])

    return demo


def main() -> None:
    demo = build_demo()
    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()