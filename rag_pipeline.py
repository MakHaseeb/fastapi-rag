"""
Shared retrieval + generation pipeline (phase 3 refactor)
=============================================================
This used to all live inside 3_ask.py. It's pulled out into its own module
for a simple, very common reason: 3_ask.py is a *command-line tool* (you
run it, it prints an answer, it exits) and 4_evaluate.py is a *batch
script* (it runs the same pipeline over 50+ questions and scores the
results) - but both need the exact same retrieval and generation logic.
Copy-pasting the funnel into both files would mean every future tweak
(a new threshold, a prompt change, a reranking fix) has to be made twice
and can silently drift out of sync between "what you test by hand" and
"what the eval script checks." One shared module, imported by both,
makes that impossible by construction.

This is also *why* it's a plain .py file and not e.g. `3_ask_lib.py`:
Python module names can't start with a digit, so `import 3_ask` was never
actually possible - which is as good a reason as any to give the shared
logic its own clearly-named home.

Nothing about the retrieval/generation behavior changed in this refactor -
it's the same three-stage funnel (vector + BM25 -> RRF fusion -> cross-
encoder rerank with a score floor) as before, just relocated.
"""

import json
import os
import pickle
import re
from pathlib import Path

import chromadb
import requests
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from prompts import SYSTEM_PROMPT

load_dotenv(Path(__file__).parent / ".env")

CHUNKS_FILE = Path(__file__).parent / "chunks.json"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
BM25_FILE = Path(__file__).parent / "bm25_index.pkl"
COLLECTION_NAME = "fastapi_tutorial_docs"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"

VECTOR_N = 15           # candidates pulled from vector search
BM25_N = 15              # candidates pulled from BM25 search
RRF_K = 60               # standard RRF damping constant
RERANK_CANDIDATES = 10   # how many fused candidates go to the cross-encoder
FINAL_K = 5              # the MAXIMUM that end up in the prompt to Claude

# The cross-encoder outputs a raw logit, not a probability - roughly,
# positive means "this chunk is actually relevant to the question" and
# negative means the model thinks it isn't. Without this floor, the code
# used to always take exactly FINAL_K candidates no matter how the
# cross-encoder scored them, which could force in a chunk the reranker
# had just judged irrelevant, purely to fill out 5 slots.
MIN_RERANK_SCORE = 0.0

# Safety net for the floor above. The eval set caught a real case: a
# question with a strong VECTOR match (0.77 cosine similarity - a good
# semantic fit) where every single reranked candidate still scored below
# MIN_RERANK_SCORE, so the floor dropped everything and the system
# returned no answer at all. The cross-encoder is a second opinion, not
# an infallible one - when the first-stage signal was already confident,
# a total wipeout is more likely the cross-encoder being overly harsh on
# this specific question than the corpus genuinely lacking an answer. In
# that specific situation (and ONLY that situation - a weak vector score
# still gets nothing, same as before), fall back to the top
# FALLBACK_TOP_K cross-encoder-RANKED candidates anyway, ignoring the
# floor. Relative ranking among candidates is still meaningful even when
# every score happens to be negative.
FALLBACK_MIN_VECTOR_SCORE = 0.6
FALLBACK_TOP_K = 2

# Gate on the top VECTOR result's raw cosine similarity, before doing any
# BM25/fusion/rerank work - cheap to check first, and a good proxy for
# "is this question even in the ballpark of this corpus at all."
MIN_SIMILARITY = 0.35


def bm25_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


_embed_model = None
_cross_encoder = None
_chunks_by_id = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embed_model


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    return _cross_encoder


def get_chunks_by_id() -> dict:
    global _chunks_by_id
    if _chunks_by_id is None:
        chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
        _chunks_by_id = {c["chunk_id"]: c for c in chunks}
    return _chunks_by_id


def vector_search(question: str, n: int) -> list[tuple[str, float]]:
    """Returns [(chunk_id, cosine_similarity), ...] ranked best-first."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    query_embedding = get_embed_model().encode([question], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=query_embedding, n_results=n)
    return [
        (result["ids"][0][i], 1 - result["distances"][0][i])
        for i in range(len(result["ids"][0]))
    ]


def bm25_search(question: str, n: int) -> list[tuple[str, float]]:
    """Returns [(chunk_id, bm25_score), ...] ranked best-first."""
    with open(BM25_FILE, "rb") as f:
        data = pickle.load(f)
    bm25: BM25Okapi = data["bm25"]
    chunk_ids: list[str] = data["chunk_ids"]

    scores = bm25.get_scores(bm25_tokenize(question))
    ranked = sorted(range(len(chunk_ids)), key=lambda i: scores[i], reverse=True)[:n]
    return [(chunk_ids[i], float(scores[i])) for i in ranked]


def reciprocal_rank_fusion(
    vector_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Combine two ranked lists into one by rank position, not raw score -
    avoids ever needing to compare a cosine similarity to a BM25 score."""
    fused_scores: dict[str, float] = {}
    for rank, (chunk_id, _score) in enumerate(vector_results):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, (chunk_id, _score) in enumerate(bm25_results):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    return sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)


def rerank(question: str, candidate_ids: list[str]) -> list[dict]:
    chunks_by_id = get_chunks_by_id()
    candidates = [chunks_by_id[cid] for cid in candidate_ids if cid in chunks_by_id]

    pairs = [(question, c["text"]) for c in candidates]
    scores = get_cross_encoder().predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda cs: cs[1], reverse=True)
    return [{**c, "rerank_score": float(s)} for c, s in ranked]


def hybrid_retrieve(question: str, debug: bool = False) -> tuple[list[dict], float]:
    vec_results = vector_search(question, VECTOR_N)
    bm25_results = bm25_search(question, BM25_N)
    best_vector_score = vec_results[0][1] if vec_results else 0.0

    if debug:
        chunks_by_id = get_chunks_by_id()
        print("-- vector search (top 5) --")
        for cid, score in vec_results[:5]:
            print(f"  {score:.3f}  {chunks_by_id[cid]['title']}")
        print("-- BM25 search (top 5) --")
        for cid, score in bm25_results[:5]:
            print(f"  {score:.2f}   {chunks_by_id[cid]['title']}")

    if best_vector_score < MIN_SIMILARITY:
        return [], best_vector_score

    fused = reciprocal_rank_fusion(vec_results, bm25_results)
    candidate_ids = [cid for cid, _ in fused[:RERANK_CANDIDATES]]

    if debug:
        chunks_by_id = get_chunks_by_id()
        print("-- RRF fused candidates sent to reranker --")
        for cid, score in fused[:RERANK_CANDIDATES]:
            print(f"  {score:.4f}  {chunks_by_id[cid]['title']}")

    all_reranked = rerank(question, candidate_ids)

    if debug:
        print("-- cross-encoder scores for every candidate --")
        for r in all_reranked:
            mark = "" if r["rerank_score"] >= MIN_RERANK_SCORE else "  (below floor, dropped)"
            print(f"  {r['rerank_score']:.3f}  {r['title']}{mark}")

    # Keep only candidates the cross-encoder actually judged relevant, up to
    # FINAL_K of them - dropping the rest even if that leaves fewer than
    # FINAL_K chunks (or, in principle, zero) in the final context.
    reranked = [r for r in all_reranked if r["rerank_score"] >= MIN_RERANK_SCORE][:FINAL_K]

    used_fallback = False
    if not reranked and all_reranked and best_vector_score >= FALLBACK_MIN_VECTOR_SCORE:
        # The floor dropped every candidate, but the very first retrieval
        # stage was confident this question belongs in this corpus at all -
        # trust the cross-encoder's RANKING (not its absolute score) enough
        # to use its top couple of picks rather than answering nothing.
        reranked = all_reranked[:FALLBACK_TOP_K]
        used_fallback = True

    if debug:
        if used_fallback:
            print(
                f"-- floor dropped everything, but vector score {best_vector_score:.3f} "
                f">= {FALLBACK_MIN_VECTOR_SCORE} -> falling back to top {FALLBACK_TOP_K} "
                f"cross-encoder-ranked candidates anyway --"
            )
        print("-- final, sent to Claude --")
        for r in reranked:
            print(f"  {r['rerank_score']:.3f}  {r['title']}")
        print()

    return reranked, best_vector_score


def build_user_prompt(question: str, retrieved: list[dict]) -> str:
    sources_block = "\n\n".join(
        f"[{i + 1}] (from \"{r['title']}\" - {r['url']})\n{r['text']}"
        for i, r in enumerate(retrieved)
    )
    return f"""Question: {question}

Source excerpts:
{sources_block}

Answer the question using only the excerpts above, with citations."""


def check_citations(answer: str, retrieved: list[dict], overlap_threshold: float = 0.3) -> list[dict]:
    chunk_by_number = {i + 1: r for i, r in enumerate(retrieved)}

    def content_words(text: str) -> set[str]:
        return set(w.lower() for w in re.findall(r"[a-zA-Z0-9]{3,}", text))

    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    flags = []
    for sentence in sentences:
        citation_numbers = [int(n) for n in re.findall(r"\[(\d+)\]", sentence)]
        if not citation_numbers:
            continue
        sentence_words = content_words(sentence)
        if not sentence_words:
            continue
        for n in citation_numbers:
            chunk = chunk_by_number.get(n)
            if not chunk:
                flags.append({"sentence": sentence, "citation": n, "reason": "citation doesn't match any retrieved source"})
                continue
            chunk_words = content_words(chunk["text"])
            overlap = len(sentence_words & chunk_words) / len(sentence_words)
            if overlap < overlap_threshold:
                flags.append({"sentence": sentence, "citation": n, "reason": f"only {overlap:.0%} word overlap with cited chunk"})
    return flags


def call_claude(user_prompt: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set - copy your .env from mini-rag or create a new one here.")

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": 700,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_prompt}],
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return "".join(block["text"] for block in data["content"] if block["type"] == "text")
