# FastAPI Docs RAG

A real RAG system over FastAPI's official tutorial documentation - same
three-stage shape as mini-rag, upgraded to the production-grade pieces
your original project brief asked for: token-based chunking, real neural
embeddings, a real vector database, and citations that link to actual
docs pages.

## Background

This project followed an earlier practice project called **mini-rag** - a
small RAG system built over 4 short sample documents, using TF-IDF
embeddings and a pickled Python list as a makeshift vector store, to
learn the core retrieve-then-generate loop before adding production-grade
infrastructure. mini-rag isn't part of this repo - it was a learning
exercise, not something meant to be reused. Every concept it taught
(chunking, embeddings, citation-grounded generation, refusing to answer
when nothing relevant was retrieved) carried forward into this project,
just with real components in place of the toy versions. The comparison
table further down spells out exactly what changed between the two.

## Setup (one time)

```bash
python3 -m pip install -r requirements.txt
```

This will take longer than the mini-rag install - `sentence-transformers`
pulls in PyTorch, and the embedding model itself (~130MB) downloads the
first time you run step 2.

**API key**: create a `.env` file in this folder with your Anthropic API
key:

```bash
printf 'ANTHROPIC_API_KEY=your-key-here\n' > .env
```

(get a key from [console.anthropic.com](https://console.anthropic.com) if
you don't already have one)

## Run it

```bash
python3 1_ingest_and_chunk.py
python3 2_build_index.py
python3 3_ask.py "How do I add a path parameter with type validation?"
```

Step 2 will be noticeably slower than the mini-rag version - it's
embedding ~40 real documentation pages instead of 4 short ones, with a
real neural network instead of TF-IDF. Expect a minute or two, not
seconds.

## What's different from mini-rag, and why

| | mini-rag | this project |
|---|---|---|
| Chunk size | 120 words | 700 tokens (~100 overlap), measured with the same tokenizer family LLMs actually use |
| Embeddings | TF-IDF (scikit-learn) | real neural embeddings (`sentence-transformers`, BAAI/bge-small-en-v1.5) |
| Vector store | Python list + pickle | ChromaDB, a real persistent vector database |
| Citations | filename + chunk id | title + real, clickable fastapi.tiangolo.com URL |
| Corpus | 4 short toy docs | ~40 real FastAPI tutorial pages |

Everything else - the retrieval-then-generate flow, the relevance
threshold that refuses off-topic questions, the citation faithfulness
checker - carries over unchanged. That's deliberate: those parts of the
pipeline don't care what your embeddings or vector store are.

## Try it on a few real questions

```bash
python3 3_ask.py "How do I add a path parameter with type validation?"
python3 3_ask.py "What's the difference between Query and Path parameters?"
python3 3_ask.py "How do dependencies with yield work?"
python3 3_ask.py "How do I deploy FastAPI with Docker?"   # should refuse - not in tutorial/
```

That last one is a good test of the relevance threshold: `deployment/`
docs aren't part of this corpus yet, so a good system should say so
rather than answering from Claude's general knowledge of Docker.

## Phase 2: hybrid retrieval + re-ranking (done)

`2_build_index.py` now builds a second index alongside the vector one - a
BM25 keyword index (`bm25_index.pkl`), via the `rank_bm25` library. If
you already ran step 2 before this update, re-run it to build the new
index:

```bash
python3 -m pip install -r requirements.txt   # picks up rank_bm25
python3 2_build_index.py
```

`3_ask.py`'s retrieval is now a three-stage funnel instead of a single
vector search:

1. Vector search AND BM25 search each independently retrieve their own
   top 15 candidates.
2. The two ranked lists are combined with **Reciprocal Rank Fusion**
   (RRF) - it fuses by rank position rather than raw score, since cosine
   similarity and BM25 scores aren't on comparable scales.
3. The fused top 10 are re-scored by a **cross-encoder**
   (`cross-encoder/ms-marco-MiniLM-L-6-v2`), which looks at the question
   and each candidate chunk together rather than as separately-embedded
   points - slower, but more accurate, which is exactly why it only runs
   on a shortlist instead of the whole corpus.

Run with `--debug` to see all three stages laid out:

```bash
python3 3_ask.py "How do dependencies with yield work?" --debug
```

`prompts.py` now holds `SYSTEM_PROMPT` with a version tag
(`SYSTEM_PROMPT_VERSION`), instead of the prompt living inline in
`3_ask.py`.

## Phase 3: offline evaluation + CI (done)

Two new files:

- **`eval_dataset.json`** - 59 question/answer pairs covering 45 of the
  51 tutorial pages: 43 single-topic questions, 10 that genuinely need
  two different pages combined, and 6 deliberately out-of-scope questions
  (Docker deployment, async Postgres drivers, etc. - none of that is in
  this corpus) that should be refused rather than answered from Claude's
  general knowledge. **Review this file** - I drafted it from the actual
  chunk text, but you should read through it (at least the ones you find
  most important) and fix anything that looks off before trusting it as
  your quality bar. It's plain JSON, easy to hand-edit.
- **`4_evaluate.py`** - runs every question in that file through the real
  pipeline (same `rag_pipeline.py` code `3_ask.py` uses) and scores four
  things: retrieval recall (did the right page actually get retrieved),
  citation faithfulness (reusing `check_citations()`'s word-overlap
  check), answer correctness (word-overlap against the reference
  answer), and refusal accuracy (did the out-of-scope questions actually
  get refused). All four are heuristics, not proof - read
  `eval_results.json` after a run to see the actual answers behind any
  number that looks wrong.

Also new: **`rag_pipeline.py`**. The retrieval + generation logic that
used to live inside `3_ask.py` moved here, since `4_evaluate.py` needs
the exact same code path - otherwise "what you test by hand" and "what
CI checks" could quietly drift apart. `3_ask.py` is now just the
command-line wrapper around it.

Run the eval:

```bash
python3 4_evaluate.py                # full 59 questions - real API calls, real model inference, expect a few minutes
python3 4_evaluate.py --sample 15    # just the first 15, for a quick check
python3 4_evaluate.py --verbose      # print pass/fail per question as it runs
```

It exits with code `0` if every metric clears its threshold (see the
constants at the top of `4_evaluate.py`) and `1` otherwise - that exit
code is what lets a CI system treat this as a real pass/fail gate.

### Wiring it into CI

`.github/workflows/eval.yml` runs this automatically on every push and
pull request to `main` (and via a manual "Run workflow" button). It
rebuilds the indexes from the committed `chunks.json` and runs
`4_evaluate.py --sample 20` (a subset, since every question is a real
API call - see the comment at the top of the workflow file for the
reasoning), then uploads the full `eval_results.json` as a downloadable
build artifact either way.

This repo is on GitHub with CI already wired up and running. `.gitignore`
excludes `.env`, the `fastapi-source/` clone, and the regenerable
`chroma_db/`/`bm25_index.pkl` build artifacts - only `chunks.json` itself
is committed, so anyone cloning this repo can rebuild the indexes without
needing the original FastAPI docs clone. The `ANTHROPIC_API_KEY` the
workflow calls Claude with is stored as a repo secret (**Settings ->
Secrets and variables -> Actions**) rather than committed anywhere -
GitHub injects it as an environment variable only inside the workflow
run, never visible in logs or to anyone without repo admin access.

Any push (or PR) to `main` triggers the eval automatically, and you'll
see a green check or red X next to the commit on GitHub - click it to see
the per-question pass/fail, or download the `eval-results` artifact for
the full JSON detail. You can also trigger it manually any time from the
**Actions** tab -> **RAG quality eval** -> **Run workflow**.

## Still open

Expanding the corpus beyond `tutorial/` to `advanced/` and `how-to/` is
still open, as is tightening the eval metrics beyond word-overlap
heuristics (e.g. an LLM-graded faithfulness check, closer to what RAGAS
does) once you've seen how the heuristic version behaves in practice.
