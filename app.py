"""Milestone 5 grounded generation app for the NOVA RAG project.

This module connects the local retrieval engine to the Groq LLM API and wraps
it in a Gradio interface. It is designed to fail gracefully when the API key is
missing or when the network/API call fails.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import gradio as gr
from dotenv import load_dotenv
from groq import Groq

from retrieval_engine import get_chroma_collection, get_model, retrieve_chunks


MODEL_NAME = "llama-3.3-70b-versatile"
TOP_K = 5
FALLBACK_MESSAGE = "I don't have enough information on that."


# Load local environment variables first so GROQ_API_KEY is available from .env.
load_dotenv()


def _safe_text(value: object) -> str:
    return "" if value is None else str(value)


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
        retrieved_chunks = retrieve_chunks(question, top_k=TOP_K, collection=collection, model=model)
    except Exception as exc:
        return {"answer": f"Retrieval failed: {exc}", "sources": []}

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