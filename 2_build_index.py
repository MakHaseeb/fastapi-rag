"""
STEP 2: Embed + Store, now building TWO indexes (phase 2)
=============================================================
Phase 1 only had a vector index. Phase 2 adds a second, completely
different kind of search index over the same chunks:

  - The VECTOR index (Chroma + sentence-transformers, unchanged from
    phase 1) finds chunks whose MEANING is close to the question, even if
    the wording is totally different.
  - A new BM25 index (via the `rank_bm25` library) finds chunks that share
    actual KEYWORDS with the question - the same family of algorithm
    traditional search engines have used for decades. It has no idea what
    words "mean," it just counts how distinctively a chunk's words match
    the query's words.

Why bother with both? Vector search can miss an exact term a user is
searching for (e.g. a specific parameter name like `response_model_exclude`)
if it doesn't have a strong "meaning" signal. BM25 nails exact terms but
misses paraphrases ("how do I stop a field from showing up in the
response" would badly confuse BM25, but a vector model understands it
means the same thing as `response_model_exclude`). Combining both is
called HYBRID RETRIEVAL, and it's one of the most reliable upgrades you
can make to a RAG system's retrieval quality.

Run this after step 1:
    python3 2_build_index.py
"""

import json
import pickle
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path(__file__).parent / "chunks.json"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
BM25_FILE = Path(__file__).parent / "bm25_index.pkl"
COLLECTION_NAME = "fastapi_tutorial_docs"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def bm25_tokenize(text: str) -> list[str]:
    """Simple lowercase word tokenizer - BM25 just needs consistent tokens,
    not anything fancy. The same function is reused in 3_ask.py to tokenize
    the query, so both sides of the comparison are tokenized identically."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vector_index(chunks: list[dict]) -> None:
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (first run downloads it, ~130MB)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [c["text"] for c in chunks]
    print("Embedding all chunks (this can take a minute or two on CPU)...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {"doc_id": c["doc_id"], "source": c["source"], "url": c["url"], "title": c["title"]}
            for c in chunks
        ],
    )
    print(f"Stored {collection.count()} chunk embeddings in {CHROMA_DIR}")


def build_bm25_index(chunks: list[dict]) -> None:
    print("Tokenizing chunks and building the BM25 keyword index...")
    tokenized_corpus = [bm25_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_FILE, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": [c["chunk_id"] for c in chunks]}, f)
    print(f"Saved BM25 index to {BM25_FILE}")


def main():
    if not CHUNKS_FILE.exists():
        raise SystemExit("chunks.json not found - run 1_ingest_and_chunk.py first.")

    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}\n")

    build_vector_index(chunks)
    print()
    build_bm25_index(chunks)

    print("\nBoth indexes are ready - run 3_ask.py next.")


if __name__ == "__main__":
    main()
