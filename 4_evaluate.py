"""
STEP 4: Offline faithfulness evaluation (phase 3)
=====================================================
This is the piece your original brief called out specifically: an offline
script that runs a curated set of question/answer pairs through the real
pipeline and scores the results, so that a change which quietly makes
answers worse - a prompt edit, a retrieval tweak, a new corpus - gets
caught automatically instead of by you happening to notice during manual
testing.

It scores four things per question, using the SAME hybrid_retrieve() /
call_claude() / check_citations() functions 3_ask.py uses (imported from
rag_pipeline.py) - this eval is only meaningful if it's testing the exact
code path you actually ship, not a separate copy of it:

  1. RETRIEVAL RECALL - for questions with a known expected source page,
     did retrieval actually surface it? This isolates retrieval quality
     from generation quality: if this is low, the bug is in vector
     search / BM25 / fusion / reranking, not in Claude's answer.

  2. CITATION FAITHFULNESS - reuses check_citations()'s word-overlap
     heuristic from 3_ask.py: of the sentences that cite a source, how
     many actually overlap with what that source says? This is a cheap
     proxy for "is the model making things up," not proof of it - a real
     production system would eventually want something like RAGAS's
     LLM-graded faithfulness metric here instead.

  3. ANSWER CORRECTNESS - another word-overlap heuristic, this time
     between the generated answer and a hand-written reference answer.
     Also a proxy, not a proof - two answers can be worded completely
     differently and both be correct, so a low score here is a prompt to
     go read the actual answer, not an automatic verdict.

  4. REFUSAL ACCURACY - the eval set includes a handful of questions that
     are deliberately NOT answerable from this corpus (e.g. Docker
     deployment, which isn't in the tutorial/ docs). A good system should
     refuse these rather than answering from Claude's general knowledge.
     This checks that it does.

All four are heuristics you can see and reason about, not a black box -
which is deliberately the same philosophy as check_citations() in
3_ask.py. Read eval_results.json after a run to see exactly why any
question failed.

Usage:
    python3 4_evaluate.py                  # full eval set
    python3 4_evaluate.py --sample 15      # first 15 questions only (fast, for CI)
    python3 4_evaluate.py --verbose        # print per-question detail as it runs

Exit code is 0 if every aggregate metric clears its threshold, 1
otherwise - that exit code is the whole reason this can "fail a build."
"""

from __future__ import annotations  # lets `str | None` work on Python 3.9, not just 3.10+

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from prompts import SYSTEM_PROMPT_VERSION
from rag_pipeline import build_user_prompt, call_claude, check_citations, hybrid_retrieve

EVAL_DATASET_FILE = Path(__file__).parent / "eval_dataset.json"
RESULTS_FILE = Path(__file__).parent / "eval_results.json"

# Aggregate pass/fail thresholds. These are the numbers a CI run actually
# checks - tune them as you learn what "good" looks like for this system.
# They're deliberately not set to 1.0: the underlying checks are word-
# overlap heuristics, not exact-match, so some slack is expected even from
# a genuinely working system.
MIN_RETRIEVAL_RECALL = 0.80
MIN_CITATION_FAITHFULNESS = 0.85
MIN_ANSWER_CORRECTNESS = 0.55
MIN_REFUSAL_ACCURACY = 0.80

REFUSAL_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "doesn't contain enough information",
    "does not contain enough information",
    "excerpts do not contain",
    "excerpts don't contain",
    "not contained in",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "no information about",
    "not covered in",
    "not mentioned in the",
]


def content_words(text: str) -> set[str]:
    return set(w.lower() for w in re.findall(r"[a-zA-Z0-9]{3,}", text))


def looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


def answer_correctness(answer: str, reference_answer: str) -> float:
    """What fraction of the reference answer's key words show up in the
    generated answer. A recall-style heuristic - the generated answer can
    say more than the reference and still score well, but it must actually
    cover the reference's key facts."""
    ref_words = content_words(reference_answer)
    if not ref_words:
        return 1.0
    answer_words = content_words(answer)
    return len(ref_words & answer_words) / len(ref_words)


def retrieval_recall(expected_titles: list[str], retrieved: list[dict], category: str) -> float:
    """For multi_topic questions, the question genuinely needs BOTH pages'
    facts combined, so recall is the fraction of expected titles actually
    found. For single_topic questions, expected_titles can list more than
    one page when the same fact is legitimately documented in more than
    one place in the corpus (FastAPI's tutorial repeats itself a fair
    amount) - any one of them showing up counts as a full hit, since the
    question only needed one correct source, not all of them."""
    if not expected_titles:
        return 1.0
    retrieved_titles = {r["title"] for r in retrieved}
    hits = sum(1 for t in expected_titles if t in retrieved_titles)
    if category == "multi_topic":
        return hits / len(expected_titles)
    return 1.0 if hits > 0 else 0.0


def git_commit_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def evaluate_one(item: dict, verbose: bool) -> dict:
    question = item["question"]
    category = item["category"]
    expected_titles = item["expected_titles"]
    reference_answer = item["reference_answer"]

    result = {
        "id": item["id"],
        "question": question,
        "category": category,
        "expected_titles": expected_titles,
    }

    try:
        retrieved, best_vector_score = hybrid_retrieve(question)
        result["retrieved_titles"] = [r["title"] for r in retrieved]
        result["best_vector_score"] = round(best_vector_score, 4)

        if category == "refusal":
            if retrieved:
                # Retrieval didn't gate it out - see whether Claude still
                # refuses rather than answering from outside knowledge.
                answer = call_claude(build_user_prompt(question, retrieved))
            else:
                answer = "(refused at retrieval - below similarity threshold, no call made)"
            result["answer"] = answer
            result["refused"] = (not retrieved) or looks_like_refusal(answer)
            result["passed"] = result["refused"]
        else:
            result["retrieval_recall"] = round(retrieval_recall(expected_titles, retrieved, category), 3)
            if not retrieved:
                result["answer"] = "(no answer - nothing cleared the retrieval threshold)"
                result["citation_faithfulness"] = 0.0
                result["answer_correctness"] = 0.0
                result["passed"] = False
            else:
                user_prompt = build_user_prompt(question, retrieved)
                answer = call_claude(user_prompt)
                flags = check_citations(answer, retrieved)
                num_citations = len(re.findall(r"\[\d+\]", answer))
                faithfulness = 1.0 if num_citations == 0 else max(0.0, 1 - len(flags) / num_citations)

                result["answer"] = answer
                result["reference_answer"] = reference_answer
                result["citation_faithfulness"] = round(faithfulness, 3)
                result["citation_flags"] = flags
                result["answer_correctness"] = round(answer_correctness(answer, reference_answer), 3)
                result["passed"] = (
                    result["retrieval_recall"] >= 0.5
                    and result["citation_faithfulness"] >= 0.5
                    and result["answer_correctness"] >= 0.4
                )
    except Exception as exc:  # keep one bad question from killing the whole run
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["passed"] = False

    if verbose:
        status = "PASS" if result.get("passed") else "FAIL"
        print(f"  [{status}] {item['id']} ({category}): {question}")
        if "error" in result:
            print(f"         error: {result['error']}")

    return result


def summarize(results: list[dict]) -> dict:
    def avg(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 3) if values else None

    topic_results = [r for r in results if r["category"] in ("single_topic", "multi_topic")]
    refusal_results = [r for r in results if r["category"] == "refusal"]
    errored = [r for r in results if "error" in r]

    summary = {
        "total_questions": len(results),
        "errored_questions": len(errored),
        "retrieval_recall_avg": avg([r.get("retrieval_recall") for r in topic_results]),
        "citation_faithfulness_avg": avg([r.get("citation_faithfulness") for r in topic_results]),
        "answer_correctness_avg": avg([r.get("answer_correctness") for r in topic_results]),
        "refusal_accuracy": avg([1.0 if r.get("refused") else 0.0 for r in refusal_results]),
        "questions_passed": sum(1 for r in results if r.get("passed")),
        "questions_failed": sum(1 for r in results if not r.get("passed")),
    }

    thresholds_met = {
        "retrieval_recall": summary["retrieval_recall_avg"] is None or summary["retrieval_recall_avg"] >= MIN_RETRIEVAL_RECALL,
        "citation_faithfulness": summary["citation_faithfulness_avg"] is None or summary["citation_faithfulness_avg"] >= MIN_CITATION_FAITHFULNESS,
        "answer_correctness": summary["answer_correctness_avg"] is None or summary["answer_correctness_avg"] >= MIN_ANSWER_CORRECTNESS,
        "refusal_accuracy": summary["refusal_accuracy"] is None or summary["refusal_accuracy"] >= MIN_REFUSAL_ACCURACY,
    }
    summary["thresholds_met"] = thresholds_met
    summary["overall_pass"] = all(thresholds_met.values())
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="only run the first N questions (faster, e.g. for CI)")
    parser.add_argument("--verbose", "-v", action="store_true", help="print per-question pass/fail as it runs")
    args = parser.parse_args()

    if not EVAL_DATASET_FILE.exists():
        raise SystemExit(f"{EVAL_DATASET_FILE} not found.")

    dataset = json.loads(EVAL_DATASET_FILE.read_text(encoding="utf-8"))
    if args.sample:
        dataset = dataset[: args.sample]

    print(f"Running {len(dataset)} eval questions against prompt version '{SYSTEM_PROMPT_VERSION}'...")
    print("(this calls the real embedding model, cross-encoder, and Claude API for each question - expect a minute or more)\n")

    start = time.time()
    results = [evaluate_one(item, args.verbose) for item in dataset]
    elapsed = time.time() - start

    summary = summarize(results)
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": SYSTEM_PROMPT_VERSION,
        "git_commit": git_commit_sha(),
        "elapsed_seconds": round(elapsed, 1),
        "summary": summary,
        "results": results,
    }
    RESULTS_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    print(f"  Questions:              {summary['total_questions']}  ({summary['questions_passed']} passed, {summary['questions_failed']} failed, {summary['errored_questions']} errored)")
    print(f"  Retrieval recall:       {summary['retrieval_recall_avg']}  (min {MIN_RETRIEVAL_RECALL})")
    print(f"  Citation faithfulness:  {summary['citation_faithfulness_avg']}  (min {MIN_CITATION_FAITHFULNESS})")
    print(f"  Answer correctness:     {summary['answer_correctness_avg']}  (min {MIN_ANSWER_CORRECTNESS})")
    print(f"  Refusal accuracy:       {summary['refusal_accuracy']}  (min {MIN_REFUSAL_ACCURACY})")
    print(f"  Elapsed:                {round(elapsed, 1)}s")
    print(f"\nFull per-question detail written to {RESULTS_FILE.name}")

    if summary["overall_pass"]:
        print("\n✅ All thresholds met.")
        sys.exit(0)
    else:
        failed = [k for k, v in summary["thresholds_met"].items() if not v]
        print(f"\n❌ Below threshold on: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
