"""
STEP 3: Ask a question (CLI)
================================
The actual retrieval + generation pipeline now lives in rag_pipeline.py,
shared with 4_evaluate.py (see that file's module docstring for why). This
file is just the command-line interface on top of it: parse the question
and flags, call the shared pipeline, print the result.

Usage:
    python3 3_ask.py "How do dependencies with yield work?"
    python3 3_ask.py "How do dependencies with yield work?" --debug
"""

import sys

from prompts import SYSTEM_PROMPT_VERSION
from rag_pipeline import build_user_prompt, call_claude, check_citations, hybrid_retrieve


def main():
    args = sys.argv[1:]
    debug = "--debug" in args or "-v" in args
    args = [a for a in args if a not in ("--debug", "-v")]

    if not args:
        raise SystemExit('Usage: python3 3_ask.py "your question here"  [--debug]')
    question = args[0]

    if debug:
        print(f'Question: "{question}"')
        print(f"(prompt version: {SYSTEM_PROMPT_VERSION})\n")

    retrieved, best_vector_score = hybrid_retrieve(question, debug=debug)

    if not retrieved:
        print(f"I don't have enough information in these docs to answer that (best match score {best_vector_score:.3f}).")
        return

    user_prompt = build_user_prompt(question, retrieved)

    if debug:
        print("Asking Claude to answer using only those chunks...\n")

    answer = call_claude(user_prompt)
    flags = check_citations(answer, retrieved)

    print(f"Q: {question}\n")
    print(f"A: {answer}\n")
    print("Sources:")
    for i, r in enumerate(retrieved):
        print(f"  [{i + 1}] {r['title']} -> {r['url']}")

    if flags:
        print(f"\n⚠ {len(flags)} citation(s) may need a second look (run with --debug for details).")

    if debug:
        print("\n" + "=" * 60)
        print("CITATION CHECK (word-overlap heuristic, not proof)")
        print("=" * 60)
        if not flags:
            print("No low-overlap citations found.")
        else:
            for f in flags:
                print(f"  ⚠ citation [{f['citation']}]: {f['reason']}")
                print(f"    sentence: \"{f['sentence']}\"")


if __name__ == "__main__":
    main()
