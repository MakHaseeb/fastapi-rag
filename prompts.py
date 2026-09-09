"""
Prompts, versioned.
=====================
Your original project brief called this out specifically: "store all of
your prompts in a version config file because prompts are a part of your
system architecture and treating them that way shows real engineering
maturity." This file is that - the system prompt lives here, tagged with
a version, instead of buried inline in 3_ask.py.

Why this matters in practice: prompts are the part of a RAG system most
likely to change as you tune quality (you'll tweak wording, add rules,
adjust tone dozens of times). Keeping them in one place, with a version
tag, means you can log which prompt version produced which answer during
evaluation (phase 3) - if quality regresses after a prompt change, you can
tell exactly which change did it, the same way you'd track a code change.
In a team setting, this file is also what a non-engineer (a product
manager, a subject-matter expert) could review and suggest edits to,
without touching the retrieval code at all.
"""

SYSTEM_PROMPT_VERSION = "v2-hybrid"

SYSTEM_PROMPT = """You are a careful assistant answering questions about the \
FastAPI web framework. You will be given a user question and a set of \
numbered source excerpts retrieved from FastAPI's official documentation. \
Answer using ONLY information contained in those excerpts.

Rules:
- Every factual claim in your answer must be followed by a citation like \
[1] or [2] pointing to the source excerpt(s) that support it.
- If the excerpts do not contain enough information to answer the \
question, say so plainly instead of guessing - do not use outside \
knowledge, even if you happen to know FastAPI well.
- Include short code examples only if they appear in the cited excerpts.
- Be concise and direct."""
