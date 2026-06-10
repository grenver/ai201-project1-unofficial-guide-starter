"""Milestone 4 retrieval engine for the NOVA RAG project.

This script provides three core capabilities:
- persistent ChromaDB collection initialization,
- embedding + upsert of chunk objects, and
- query-time retrieval with structured, inspectable output.

It also includes a small mock chunk corpus so the retrieval path can be
exercised immediately before wiring in the Milestone 3 ingestion pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import chromadb
from sentence_transformers import SentenceTransformer


DEFAULT_PERSIST_DIR = Path("./chroma_db")
DEFAULT_COLLECTION_NAME = "nova_cs_guide"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.5


@dataclass
class ChunkRecord:
    """Simple chunk representation compatible with Milestone 3 output."""

    text: str
    metadata: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate and query the NOVA ChromaDB collection.")
    parser.add_argument(
        "--persist-dir",
        default=str(DEFAULT_PERSIST_DIR),
        help="Directory used for persistent ChromaDB storage.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the existing collection contents before loading mock chunks.",
    )
    parser.add_argument(
        "--skip-seed",
        action="store_true",
        help="Skip loading the built-in mock chunk data before running validation queries.",
    )
    return parser.parse_args()


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def sanitize_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """ChromaDB metadata must remain flat and primitive-valued."""

    if not metadata:
        return {}

    sanitized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = coerce_text(value)
    return sanitized


def normalize_chunk_object(chunk: Any) -> ChunkRecord:
    """Accept dicts or simple objects with `text` and `metadata` attributes."""

    if isinstance(chunk, dict):
        text = coerce_text(chunk.get("text", ""))
        metadata = chunk.get("metadata", {})
    else:
        text = coerce_text(getattr(chunk, "text", ""))
        metadata = getattr(chunk, "metadata", {})

    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": coerce_text(metadata)}

    return ChunkRecord(text=text, metadata=sanitize_metadata(metadata))


def get_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def get_chroma_collection(
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def clear_collection(collection) -> None:
    """Reset collection contents without dropping the persisted directory."""

    try:
        collection.delete(where={})
    except Exception:
        # If the collection is already empty or the backend rejects empty deletes,
        # we continue and let the next upsert overwrite by id.
        pass


def _build_id(chunk: ChunkRecord, index: int) -> str:
    base = chunk.metadata.get("chunk_id") or chunk.metadata.get("comment_id") or chunk.metadata.get("post_id")
    if base:
        return f"{base}-{index}"
    source_file = chunk.metadata.get("source_file") or "chunk"
    return f"{source_file}-{index}"


def populate_vector_db(
    chunks: Sequence[Any],
    collection=None,
    model: Optional[SentenceTransformer] = None,
) -> int:
    """Embed and upsert a chunk list into ChromaDB.

    Parameters
    ----------
    chunks:
        A list of chunk-like objects. Each item must provide `text` and `metadata`.
    collection:
        Optional prebuilt ChromaDB collection. If omitted, a default persistent
        collection is created.
    model:
        Optional sentence-transformers model. If omitted, the default model is loaded.

    Returns
    -------
    int
        The number of valid chunks inserted into the vector store.
    """

    collection = collection or get_chroma_collection()
    model = model or get_model()

    normalized_chunks = [normalize_chunk_object(chunk) for chunk in chunks]
    normalized_chunks = [chunk for chunk in normalized_chunks if chunk.text.strip()]
    if not normalized_chunks:
        return 0

    texts = [chunk.text for chunk in normalized_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()
    ids = [_build_id(chunk, index) for index, chunk in enumerate(normalized_chunks)]
    metadatas = [sanitize_metadata(chunk.metadata) for chunk in normalized_chunks]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(normalized_chunks)


def _format_result(item: Dict[str, Any]) -> Dict[str, Any]:
    distance = item.get("distance")
    similarity = None
    if distance is not None:
        similarity = 1.0 - float(distance)
    metadata = item.get("metadata") or {}
    return {
        "text": item.get("text", ""),
        "metadata": metadata,
        "distance": distance,
        "similarity": similarity,
        "source_url": metadata.get("source_url") or metadata.get("url") or "",
        "file_reference": metadata.get("source_file") or metadata.get("file_reference") or "",
    }


def retrieve_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    collection=None,
    model: Optional[SentenceTransformer] = None,
) -> List[Dict[str, Any]]:
    """Return the nearest chunks for a natural-language query."""

    collection = collection or get_chroma_collection()
    model = model or get_model()

    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    structured_results: List[Dict[str, Any]] = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        structured_results.append(
            _format_result(
                {
                    "text": text,
                    "metadata": metadata or {},
                    "distance": distance,
                }
            )
        )

    return structured_results


def print_retrieval_results(query: str, results: Sequence[Dict[str, Any]], weak_threshold: float = DEFAULT_THRESHOLD) -> None:
    print(f"\n=== Query ===\n{query}")
    if not results:
        print("No results found.")
        return

    for rank, item in enumerate(results, start=1):
        distance = item.get("distance")
        weak_flag = " [WEAK MATCH]" if isinstance(distance, (int, float)) and distance > weak_threshold else ""
        print(f"\n--- Rank {rank}{weak_flag} ---")
        print(f"distance: {distance}")
        print(f"similarity: {item.get('similarity')}")
        print("attribution:")
        attribution = {
            "source_url": item.get("source_url"),
            "file_reference": item.get("file_reference"),
        }
        print(attribution)
        print("metadata:")
        print(item.get("metadata", {}))
        print("text:")
        print(item.get("text", ""))


def mock_chunks() -> List[ChunkRecord]:
    """Small seed corpus for immediate retrieval smoke-testing."""

    return [
        ChunkRecord(
            text=(
                "THREAD TITLE: CSC 205 vs CSC 215\n\n"
                "CURRENT COMMENT: CSC 205 focuses on Computer Organization and aligns with GMU's computer science requirements, "
                "while CSC 215 focuses on Computer Systems and is typically required for Virginia Tech transfers."
            ),
            metadata={
                "source_url": "https://www.reddit.com/r/nvcc/comments/1nrcl22/difference_between_csc_205_and_215/",
                "source_file": "thread_eval_1.json",
                "source_index": 101,
                "post_id": "1nrcl22",
                "post_title": "difference between csc 205 and 215",
                "chunk_type": "comment",
                "comment_id": "eval205",
                "comment_depth": 0,
            },
        ),
        ChunkRecord(
            text=(
                "THREAD TITLE: Computer Science transfer to GMU\n\n"
                "CURRENT COMMENT: Students emphasize utilizing the ADVANCE pathway program to guarantee credit matching, "
                "ensuring math sequences like Discrete Math and Calculus are completed before transferring to avoid graduation delays."
            ),
            metadata={
                "source_url": "https://www.reddit.com/r/nvcc/comments/1py0foc/computer_science_transfer_to_gmu/",
                "source_file": "thread_eval_2.json",
                "source_index": 102,
                "post_id": "1py0foc",
                "post_title": "computer science transfer to GMU",
                "chunk_type": "comment",
                "comment_id": "evalgmu",
                "comment_depth": 0,
            },
        ),
        ChunkRecord(
            text=(
                "THREAD TITLE: PHYS 102 vs 201 vs 231\n\n"
                "CURRENT COMMENT: University Physics (PHYS 231/232) is calculus-based and mandatory for Engineering and rigorous CS degrees, "
                "whereas College Physics (PHYS 201/202) is algebra-based and will not satisfy engineering transfer agreements."
            ),
            metadata={
                "source_url": "https://www.reddit.com/r/nvcc/comments/pe9h81/phys_102_vs_201_vs_231/",
                "source_file": "thread_eval_3.json",
                "source_index": 103,
                "post_id": "pe9h81",
                "post_title": "PHYS 102 vs 201 vs 231",
                "chunk_type": "comment",
                "comment_id": "evalphys",
                "comment_depth": 0,
            },
        ),
        ChunkRecord(
            text=(
                "THREAD TITLE: Please advise me regarding CSC-221 course\n\n"
                "ORIGINAL POST: I plan to take the online CS50 course by Harvard during this term as a preparation for CSC-221. "
                "Please tell me if it is a good plan."
            ),
            metadata={
                "source_url": "https://www.reddit.com/r/nvcc/comments/xcujjw/please_advise_me_regarding_csc221_course/",
                "source_file": "thread1.json",
                "source_index": 1,
                "post_id": "xcujjw",
                "post_title": "Please advise me regarding CSC-221 course",
                "chunk_type": "post",
            },
        ),
        ChunkRecord(
            text=(
                "THREAD TITLE: Csc 223?\n\n"
                "CURRENT COMMENT: I had a hard time adjusting to the difficulty of the coursework at UVA after transferring, so make sure your work ethic is at a good level because the difficulty will jump once you transfer."
            ),
            metadata={
                "source_url": "https://www.reddit.com/r/nvcc/comments/1d3qzwx/csc_223/",
                "source_file": "thread2.json",
                "source_index": 2,
                "post_id": "1d3qzwx",
                "post_title": "Csc 223?",
                "chunk_type": "comment",
                "comment_id": "l6qdpvj",
                "comment_depth": 2,
            },
        ),
        ChunkRecord(
            text=(
                "THREAD TITLE: Csc 223?\n\n"
                "CURRENT COMMENT: if you’ve already taken an equivalent, put that in and maybe a word document explaining your situation. it'll take a few days to get reviewed but that was enough for me"
            ),
            metadata={
                "source_url": "https://www.reddit.com/r/nvcc/comments/1d3qzwx/csc_223/",
                "source_file": "thread2.json",
                "source_index": 2,
                "post_id": "1d3qzwx",
                "post_title": "Csc 223?",
                "chunk_type": "comment",
                "comment_id": "l6qdpvj_1",
                "comment_depth": 3,
            },
        ),
        ChunkRecord(
            text=(
                "THREAD TITLE: Transfer from A.S. Engineering to CS\n\n"
                "CURRENT COMMENT: Use the ADVANCE pathway when possible and make sure your math sequence is complete before transfer to avoid graduation delays."
            ),
            metadata={
                "source_url": "https://www.reddit.com/r/nvcc/comments/1t92mmm/transfer_from_as_engineering_to_cs_degree_at_va/",
                "source_file": "thread9.json",
                "source_index": 9,
                "post_id": "1t92mmm",
                "post_title": "Transfer from A.S. Engineering to CS",
                "chunk_type": "comment",
                "comment_id": "adv1",
                "comment_depth": 0,
            },
        ),
        ChunkRecord(
            text=(
                "THREAD TITLE: PHYS 232 with Prof Medvar\n\n"
                "CURRENT COMMENT: University Physics II is calculus-based and the labs/exams are practical realities students should expect when transferring into engineering-heavy programs."
            ),
            metadata={
                "source_url": "https://www.reddit.com/r/nvcc/comments/uzgmh7/phys_232_with_prof_medvar/",
                "source_file": "thread8.json",
                "source_index": 8,
                "post_id": "uzgmh7",
                "post_title": "PHYS 232 with Prof Medvar",
                "chunk_type": "comment",
                "comment_id": "phys1",
                "comment_depth": 0,
            },
        ),
    ]


def ensure_seed_data(collection, model, reset: bool = False) -> None:
    if reset:
        clear_collection(collection)
    populate_vector_db(mock_chunks(), collection=collection, model=model)


def main() -> None:
    args = parse_args()
    persist_dir = Path(args.persist_dir)
    collection = get_chroma_collection(persist_dir=persist_dir)
    model = get_model()

    if not args.skip_seed:
        ensure_seed_data(collection, model, reset=args.reset)

    test_queries = [
        "What do students say is the difference between CSC 205 and CSC 215?",
        "What advice is given for transferring from NOVA CS to GMU?",
        "Which physics sequence do students recommend for engineering-oriented transfer, and when should each option be used?",
    ]

    for query in test_queries:
        results = retrieve_chunks(query, top_k=DEFAULT_TOP_K, collection=collection, model=model)
        print_retrieval_results(query, results, weak_threshold=DEFAULT_THRESHOLD)


if __name__ == "__main__":
    main()