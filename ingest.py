"""Milestone 3 ingestion and chunking pipeline for the NOVA RAG corpus.

This script:
- scans every JSON file in the documents/ directory,
- parses Reddit Listing exports,
- preserves thread context by prepending the title and original post body into every chunk,
- recursively chunks text using a natural-break-first strategy,
- filters malformed or empty chunks, and
- prints a summary plus 5 representative chunks for inspection.

The implementation uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_CHUNK_SIZE = 1200
DEFAULT_OVERLAP = 250
DEFAULT_SAMPLE_COUNT = 5
DEFAULT_RANDOM_SEED = 42


WHITESPACE_RE = re.compile(r"\s+")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
COMMON_BOILERPLATE_RE = re.compile(
    r"(I am a bot, and this action was performed automatically\.|"
    r"Please contact the moderators of this subreddit if you have any questions or concerns\.)",
    re.IGNORECASE,
)


@dataclass
class ThreadPost:
    """Normalized representation of a Reddit thread post."""

    title: str
    selftext: str
    url: str
    source_file: str
    source_index: int
    post_id: str
    created_utc: Optional[float] = None
    subreddit: Optional[str] = None
    permalink: Optional[str] = None
    author: Optional[str] = None
    num_comments: Optional[int] = None


@dataclass
class ChunkRecord:
    """A chunk of text plus its metadata."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Reddit JSON exports into chunks.")
    parser.add_argument(
        "--documents-dir",
        default="documents",
        help="Directory containing Reddit JSON exports (default: documents).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Target maximum chunk size in characters.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="Desired overlap between adjacent chunks in characters.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help="How many representative chunks to print.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for representative chunk sampling.",
    )
    return parser.parse_args()


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def coerce_text(value: Any) -> str:
    """Best-effort conversion of a JSON value to readable text."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("name", "display_name", "title", "url", "id"):
            nested_value = value.get(key)
            if isinstance(nested_value, str) and nested_value.strip():
                return nested_value
        return ""
    return str(value)


def clean_text(text: Any) -> str:
    """Clean text extracted from Reddit export fields.

    The goal is to keep the semantic content while stripping HTML entities,
    markup artifacts, and boilerplate that does not help retrieval.
    """

    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    cleaned = html.unescape(text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = HTML_COMMENT_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = cleaned.replace("[deleted]", "").replace("[removed]", "")
    cleaned = COMMON_BOILERPLATE_RE.sub("", cleaned)
    cleaned = cleaned.replace("&nbsp;", " ")
    cleaned = normalize_whitespace(cleaned)
    return cleaned


def load_json_document(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_post_and_comments(document: Any) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract the main post and top-level comments from a Reddit Listing export.

    The expected shape is usually a two-item list:
    - [0] listing for the post
    - [1] listing for comments

    The function is tolerant of small structural differences.
    """

    if isinstance(document, list):
        if not document:
            return None, []

        post = None
        comments: List[Dict[str, Any]] = []

        for item in document:
            if not isinstance(item, dict):
                continue

            if item.get("kind") == "wikipage":
                data = item.get("data", {}) if isinstance(item.get("data", {}), dict) else {}
                raw_content_md = coerce_text(data.get("content_md", ""))
                content_md = clean_text(raw_content_md)
                wiki_title = None
                if raw_content_md:
                    first_heading = next((line.strip() for line in raw_content_md.splitlines() if line.strip().startswith("#")), "")
                    if first_heading:
                        wiki_title = clean_text(first_heading.lstrip("#"))

                post = {
                    "title": wiki_title or "NOVA wiki index",
                    "selftext": content_md,
                    "url": "https://www.reddit.com/r/nvcc/wiki/index/",
                    "id": data.get("revision_id", "wiki_index"),
                    "created_utc": data.get("revision_date"),
                    "subreddit": "r/nvcc",
                    "permalink": "https://www.reddit.com/r/nvcc/wiki/index/",
                    "author": data.get("revision_by"),
                    "num_comments": 0,
                }
                return post, []

            kind = item.get("kind")
            data = item.get("data", {})
            children = data.get("children", []) if isinstance(data, dict) else []

            if kind == "Listing" and children:
                first_child = children[0]
                if isinstance(first_child, dict) and first_child.get("kind") == "t3" and post is None:
                    post = first_child.get("data", {})
                    continue

                # Comment listing: collect top-level t1 children.
                for child in children:
                    if isinstance(child, dict) and child.get("kind") == "t1":
                        comments.append(child.get("data", {}))

        return post, comments

    if isinstance(document, dict):
        if document.get("kind") == "wikipage":
            data = document.get("data", {}) if isinstance(document.get("data", {}), dict) else {}
            raw_content_md = coerce_text(data.get("content_md", ""))
            content_md = clean_text(raw_content_md)
            wiki_title = None
            if raw_content_md:
                first_heading = next((line.strip() for line in raw_content_md.splitlines() if line.strip().startswith("#")), "")
                if first_heading:
                    wiki_title = clean_text(first_heading.lstrip("#"))

            post = {
                "title": wiki_title or "NOVA wiki index",
                "selftext": content_md,
                "url": "https://www.reddit.com/r/nvcc/wiki/index/",
                "id": data.get("revision_id", "wiki_index"),
                "created_utc": data.get("revision_date"),
                "subreddit": "r/nvcc",
                "permalink": "https://www.reddit.com/r/nvcc/wiki/index/",
                "author": data.get("revision_by"),
                "num_comments": 0,
            }
            return post, []

        data = document.get("data", {})
        children = data.get("children", []) if isinstance(data, dict) else []
        post = None
        comments = []

        for child in children:
            if not isinstance(child, dict):
                continue
            if child.get("kind") == "t3" and post is None:
                post = child.get("data", {})
            elif child.get("kind") == "t1":
                comments.append(child.get("data", {}))

        return post, comments

    return None, []


def normalize_comment_replies(replies: Any) -> List[Dict[str, Any]]:
    if not replies or not isinstance(replies, dict):
        return []
    data = replies.get("data", {})
    if not isinstance(data, dict):
        return []
    children = data.get("children", [])
    result: List[Dict[str, Any]] = []
    for child in children:
        if isinstance(child, dict) and child.get("kind") == "t1":
            result.append(child.get("data", {}))
    return result


def build_thread_post(post_data: Dict[str, Any], source_file: Path, source_index: int) -> ThreadPost:
    source_url = clean_text(post_data.get("url", "")) or source_file.name
    return ThreadPost(
        title=clean_text(post_data.get("title", "")),
        selftext=clean_text(post_data.get("selftext", "")),
        url=source_url,
        source_file=source_file.name,
        source_index=source_index,
        post_id=clean_text(post_data.get("id", "")),
        created_utc=post_data.get("created_utc"),
        subreddit=clean_text(post_data.get("subreddit_name_prefixed", post_data.get("subreddit", ""))) or None,
        permalink=clean_text(post_data.get("permalink", "")) or None,
        author=clean_text(coerce_text(post_data.get("author", ""))) or None,
        num_comments=post_data.get("num_comments"),
    )


def build_thread_context(post: ThreadPost) -> str:
    parts = [f"THREAD TITLE: {post.title}".strip()]
    if post.url:
        parts.append(f"THREAD URL: {post.url}")
    if post.selftext:
        parts.append(f"ORIGINAL POST:\n{post.selftext}")
    return "\n\n".join(parts).strip()


def build_comment_context(
    thread_post: ThreadPost,
    ancestor_chain: Sequence[Dict[str, Any]],
    comment_body: str,
) -> str:
    parts = [build_thread_context(thread_post)]

    if ancestor_chain:
        ancestry_lines = []
        for ancestor in ancestor_chain:
            ancestor_author = clean_text(ancestor.get("author", "")) or "unknown"
            ancestor_body = clean_text(ancestor.get("body", ""))
            if ancestor_body:
                ancestry_lines.append(f"- {ancestor_author}: {ancestor_body}")
        if ancestry_lines:
            parts.append("ANCESTOR CONTEXT:\n" + "\n".join(ancestry_lines))

    parts.append(f"CURRENT COMMENT:\n{comment_body}")
    return "\n\n".join(part for part in parts if part).strip()


def build_metadata(
    thread_post: ThreadPost,
    chunk_type: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "chunk_type": chunk_type,
        "source_url": thread_post.url,
        "source_file": thread_post.source_file,
        "source_index": thread_post.source_index,
        "post_id": thread_post.post_id,
        "post_title": thread_post.title,
        "post_created_utc": thread_post.created_utc,
        "subreddit": thread_post.subreddit,
        "permalink": thread_post.permalink,
        "author": thread_post.author,
        "num_comments": thread_post.num_comments,
    }
    if extra:
        metadata.update(extra)
    return metadata


def split_long_segment(segment: str, chunk_size: int) -> List[str]:
    """Hard split a segment that is still larger than the target size."""

    pieces: List[str] = []
    start = 0
    while start < len(segment):
        end = min(start + chunk_size, len(segment))
        pieces.append(segment[start:end].strip())
        start = end
    return [piece for piece in pieces if piece]


def recursive_split_text(text: str, chunk_size: int, separators: Optional[Sequence[str]] = None) -> List[str]:
    """Split text using natural boundaries first, then a hard cutoff.

    This function approximates a recursive character splitter without external
    dependencies.
    """

    if not text:
        return []
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    if separators is None:
        separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

    for separator in separators:
        if separator not in text:
            continue

        parts = text.split(separator)
        if len(parts) == 1:
            continue

        result: List[str] = []
        current = ""

        for index, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue

            candidate = part if not current else current + separator + part
            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                result.extend(recursive_split_text(current, chunk_size, separators[1:]))
                current = ""

            if len(part) > chunk_size:
                result.extend(recursive_split_text(part, chunk_size, separators[1:]))
            else:
                current = part

        if current:
            result.extend(recursive_split_text(current, chunk_size, separators[1:]))

        if result:
            return result

    return split_long_segment(text, chunk_size)


def merge_segments_with_overlap(segments: Sequence[str], chunk_size: int, overlap: int) -> List[str]:
    """Merge smaller segments into chunk-sized windows with overlap."""

    if not segments:
        return []

    chunks: List[str] = []
    current_segments: List[str] = []

    def current_text(parts: Sequence[str]) -> str:
        return "\n\n".join(parts).strip()

    def build_overlap_seed(parts: Sequence[str]) -> List[str]:
        if overlap <= 0 or not parts:
            return []
        seed: List[str] = []
        for part in reversed(parts):
            candidate = [part] + seed
            if len(current_text(candidate)) > overlap and seed:
                break
            seed = candidate
            if len(current_text(seed)) >= overlap:
                break
        return seed

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        tentative = current_text(current_segments + [segment])
        if current_segments and len(tentative) > chunk_size:
            chunk_text = current_text(current_segments)
            if chunk_text:
                chunks.append(chunk_text)
            current_segments = build_overlap_seed(current_segments)
            tentative = current_text(current_segments + [segment])

            if current_segments and len(tentative) > chunk_size:
                # If the overlap seed is too large, shorten it aggressively.
                while current_segments and len(current_text(current_segments)) > overlap:
                    current_segments = current_segments[1:]

        current_segments.append(segment)

        if len(current_text(current_segments)) > chunk_size:
            # The last segment itself may be larger than the target size.
            chunk_text = current_text(current_segments[:-1])
            if chunk_text:
                chunks.append(chunk_text)
            current_segments = build_overlap_seed(current_segments[:-1])
            current_segments.append(segment)

    if current_segments:
        final_text = current_text(current_segments)
        if final_text:
            chunks.append(final_text)

    return [chunk for chunk in chunks if chunk.strip()]


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    segments = recursive_split_text(text, chunk_size)
    return merge_segments_with_overlap(segments, chunk_size, overlap)


def traverse_comment_tree(
    thread_post: ThreadPost,
    comment_data: Dict[str, Any],
    chunk_size: int,
    overlap: int,
    ancestor_chain: Optional[List[Dict[str, Any]]] = None,
) -> List[ChunkRecord]:
    ancestor_chain = ancestor_chain or []

    body = clean_text(comment_data.get("body", ""))
    if not body:
        return []

    comment_context = build_comment_context(thread_post, ancestor_chain, body)
    comment_chunks = chunk_text(comment_context, chunk_size, overlap)

    comment_chunks = [chunk for chunk in comment_chunks if chunk.strip()]
    records: List[ChunkRecord] = []

    for chunk_index, chunk in enumerate(comment_chunks):
        metadata = build_metadata(
            thread_post,
            chunk_type="comment",
            extra={
                "comment_id": clean_text(comment_data.get("id", "")),
                "comment_depth": comment_data.get("depth"),
                "comment_author": clean_text(comment_data.get("author", "")) or None,
                "parent_id": clean_text(comment_data.get("parent_id", "")) or None,
                "is_submitter": comment_data.get("is_submitter"),
                "comment_index": chunk_index,
                "ancestor_count": len(ancestor_chain),
            },
        )
        records.append(ChunkRecord(text=chunk, metadata=metadata))

    replies = normalize_comment_replies(comment_data.get("replies"))
    for reply in replies:
        records.extend(traverse_comment_tree(thread_post, reply, chunk_size, overlap, ancestor_chain + [comment_data]))

    return records


def chunk_thread(post: ThreadPost, comments: Sequence[Dict[str, Any]], chunk_size: int, overlap: int) -> List[ChunkRecord]:
    records: List[ChunkRecord] = []

    # Chunk the original post while keeping the thread title bound to the body.
    post_context = build_thread_context(post)
    post_chunks = chunk_text(post_context, chunk_size, overlap)
    for chunk_index, chunk in enumerate(post_chunks):
        metadata = build_metadata(
            post,
            chunk_type="post",
            extra={
                "chunk_index": chunk_index,
                "source_kind": "submission",
            },
        )
        records.append(ChunkRecord(text=chunk, metadata=metadata))

    for comment in comments:
        records.extend(traverse_comment_tree(post, comment, chunk_size, overlap))

    return records


def ingest_documents(documents_dir: Path, chunk_size: int, overlap: int) -> List[ChunkRecord]:
    all_chunks: List[ChunkRecord] = []
    json_files = sorted(path for path in documents_dir.glob("*.json") if path.is_file())

    for source_index, json_path in enumerate(json_files, start=1):
        try:
            document = load_json_document(json_path)
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"[WARN] Skipping {json_path.name}: could not parse JSON ({exc})")
            continue

        post_data, top_level_comments = extract_post_and_comments(document)
        if not post_data:
            print(f"[WARN] Skipping {json_path.name}: no thread post found")
            continue

        post = build_thread_post(post_data, json_path, source_index)
        thread_chunks = chunk_thread(post, top_level_comments, chunk_size, overlap)
        all_chunks.extend(thread_chunks)

    return all_chunks


def filter_valid_chunks(chunks: Sequence[ChunkRecord]) -> List[ChunkRecord]:
    valid: List[ChunkRecord] = []
    for chunk in chunks:
        text = clean_text(chunk.text)
        metadata = dict(chunk.metadata)
        if not text:
            continue
        if len(text) == 0:
            continue
        valid.append(ChunkRecord(text=text, metadata=metadata))
    return valid


def print_representative_chunks(chunks: Sequence[ChunkRecord], sample_count: int, seed: int) -> None:
    if not chunks:
        print("No chunks were produced.")
        return

    rng = random.Random(seed)
    sample_size = min(sample_count, len(chunks))
    sample_indices = sorted(rng.sample(range(len(chunks)), sample_size))

    print("\n=== Representative Chunks ===")
    for display_index, chunk_index in enumerate(sample_indices, start=1):
        chunk = chunks[chunk_index]
        print(f"\n--- Sample {display_index} / {sample_size} (global index {chunk_index}) ---")
        print("METADATA:")
        print(json.dumps(chunk.metadata, indent=2, sort_keys=True, ensure_ascii=False, default=str))
        print("TEXT:")
        print(chunk.text)


def main() -> None:
    args = parse_args()
    documents_dir = Path(args.documents_dir)

    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {documents_dir}")

    raw_chunks = ingest_documents(documents_dir, args.chunk_size, args.overlap)
    chunks = filter_valid_chunks(raw_chunks)

    print(f"Total consolidated chunk count: {len(chunks)}")
    print_representative_chunks(chunks, args.sample_count, args.seed)


if __name__ == "__main__":
    main()