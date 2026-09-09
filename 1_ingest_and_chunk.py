"""
STEP 1: Ingest + Chunk (real version)
=======================================
Same job as the mini-rag version, leveled up for a real corpus:

  - Walks every .md file under the FastAPI docs' tutorial/ folder
    (recursively - some pages live in subfolders like tutorial/security/).
  - Chunks by TOKENS instead of words, using tiktoken - the same kind of
    tokenizer LLMs actually use to count length. 500-800 tokens per chunk
    with ~100 tokens of overlap, matching the sizing your original project
    brief called for (the mini version used small word-based chunks purely
    because the sample docs were tiny toy files).
  - Records a real, clickable source URL for each chunk, built from the
    file's path - so a citation can point to the actual FastAPI docs page,
    not just a filename.

Run this first:
    python3 1_ingest_and_chunk.py
"""

import json
import re
from pathlib import Path

import tiktoken

# Point this at wherever you cloned the FastAPI repo.
DOCS_ROOT = Path(__file__).parent / "fastapi-source" / "docs" / "en" / "docs"
SECTION = "tutorial"  # which top-level docs folder to ingest
OUTPUT_FILE = Path(__file__).parent / "chunks.json"

CHUNK_SIZE_TOKENS = 700
CHUNK_OVERLAP_TOKENS = 100

ENCODING = tiktoken.get_encoding("cl100k_base")


def path_to_doc_url(md_path: Path, section_root: Path) -> str:
    """Turn a local file path into the real fastapi.tiangolo.com URL for that page."""
    rel = md_path.relative_to(section_root).with_suffix("")
    slug = str(rel).replace("\\", "/")
    if slug.endswith("/index"):
        slug = slug[: -len("/index")]
    return f"https://fastapi.tiangolo.com/{slug}/"


def extract_title(markdown_text: str, fallback: str) -> str:
    """Use the first Markdown H1 (# Heading) as the page title, if there is one."""
    match = re.search(r"^#\s+(.+?)\s*(\{.*\})?\s*$", markdown_text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return fallback


def load_documents(section_root: Path) -> list[dict]:
    documents = []
    for path in sorted(section_root.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(section_root)
        documents.append(
            {
                "doc_id": str(rel.with_suffix("")).replace("/", "__"),
                "source": str(rel),
                "url": path_to_doc_url(path, section_root),
                "title": extract_title(text, fallback=path.stem),
                "text": text,
            }
        )
    return documents


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    tokens = ENCODING.encode(text)
    if not tokens:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(tokens):
        window = tokens[start : start + chunk_size]
        chunks.append(ENCODING.decode(window))
        if start + chunk_size >= len(tokens):
            break
        start += step
    return chunks


def main():
    section_root = DOCS_ROOT / SECTION
    if not section_root.exists():
        raise SystemExit(
            f"{section_root} not found. Make sure you've cloned the FastAPI repo "
            f"into fastapi-source/ next to this script first."
        )

    documents = load_documents(section_root)
    print(f"Loaded {len(documents)} document(s) from {section_root}")

    all_chunks = []
    for doc in documents:
        pieces = chunk_text(doc["text"], CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)
        for i, piece in enumerate(pieces):
            all_chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}::chunk{i}",
                    "doc_id": doc["doc_id"],
                    "source": doc["source"],
                    "url": doc["url"],
                    "title": doc["title"],
                    "text": piece,
                }
            )

    OUTPUT_FILE.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    token_counts = [len(ENCODING.encode(c["text"])) for c in all_chunks]
    print(f"Wrote {len(all_chunks)} chunks to {OUTPUT_FILE}")
    print(
        f"Chunk size: min={min(token_counts)}, max={max(token_counts)}, "
        f"avg={sum(token_counts) / len(token_counts):.0f} tokens"
    )


if __name__ == "__main__":
    main()
