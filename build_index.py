"""Build the ChromaDB index from the real ingested corpus.

This is the glue between Milestone 3 (ingestion/chunking in ``ingest.py``) and
Milestone 4 (the vector store in ``retrieval_engine.py``). It runs the ingestion
pipeline over ``documents/``, then embeds and upserts the resulting chunks into a
freshly rebuilt persistent collection so the store contains only real corpus
chunks -- no mock seed data.

Usage:
    python build_index.py
    python build_index.py --documents-dir documents --chunk-size 1200 --overlap 250
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chromadb

from ingest import filter_valid_chunks, ingest_documents
from retrieval_engine import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIR,
    get_model,
    populate_vector_db,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed the ingested corpus into ChromaDB.")
    parser.add_argument("--documents-dir", default="documents")
    parser.add_argument("--persist-dir", default=str(DEFAULT_PERSIST_DIR))
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Drop and recreate the collection so the rebuild is clean and reproducible.
    client = chromadb.PersistentClient(path=args.persist_dir)
    try:
        client.delete_collection(args.collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=args.collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    raw_chunks = ingest_documents(Path(args.documents_dir), args.chunk_size, args.overlap)
    chunks = filter_valid_chunks(raw_chunks)

    model = get_model()
    inserted = populate_vector_db(chunks, collection=collection, model=model)

    print(f"Ingested chunks: {len(chunks)}")
    print(f"Inserted into collection: {inserted}")
    print(f"Collection count: {collection.count()}")


if __name__ == "__main__":
    main()
