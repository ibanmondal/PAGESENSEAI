# Interview talking points

How to narrate this project for different roles. The same codebase, three angles.

## The 30-second pitch (any role)

> "I built a multi-tab RAG browser copilot. The interesting part isn't the
> extension — it's the retrieval. I built a hybrid retriever, wrote an eval
> harness with my own labeled set, then fine-tuned the cross-encoder re-ranker
> and measured the lift. Retrieval nDCG went from X to Y."

## GenAI / AI Engineer angle

- **Multi-agent orchestration:** LangGraph, 3 explicit nodes (retrieve → generate
  → fact-check). Each node is a pure function of state — inspectable and testable.
- **Hybrid retrieval:** BM25 + dense, fused via Reciprocal Rank Fusion (rank-based,
  not score-based — robust to incomparable score scales).
- **Re-ranking stage:** cross-encoder over top-20 → top-5. The expensive-but-accurate
  second stage every serious RAG system has.
- **Model routing:** rule-based router (the *baseline*) sends easy queries to
  Gemini Flash and hard ones to GPT-4.1. A learned classifier is the documented
  next step — A/B it against this baseline with the same harness.

Be ready to answer: *"Why RRF instead of score fusion?"* → because BM25 and cosine
scores aren't comparable without calibration; ranks are.

## ML Engineer angle (lead with this)

Frame the project as **learning-to-rank**:

- "I treated retrieval as a ranking problem. I built a labeled eval set of
  (query, relevant-doc) pairs, defined the grading protocol (doc-level binary
  relevance), and ran controlled experiments."
- Explain metric choice: **nDCG@10** (position-discounted, normalized — the IR
  standard), **MRR** (first-relevant, intuitive), **Recall@k** (does the right
  source even get surfaced — the RAG-specific one).
- Talk train/val split + hard-negative mining in `finetune_reranker.py`.
- Report before/after with the *same* test set — that's what makes it science.

Be ready to answer: *"How do you know the lift isn't noise?"* → held-out test
split, and at 150+ queries the per-query variance averages out; could add a
paired bootstrap if pushed.

## Data Scientist angle

- Lead with evaluation methodology: metric selection, grading protocol,
  train/holdout discipline, class imbalance in relevance labels (most pairs are
  negatives — that's why hard-negative mining matters).
- "Citation faithfulness" as a future metric: an LLM-as-judge that scores whether
  each `[#]` actually supports its claim — a precision-of-grounding measure.
- Latency/cost trade-off as an experimentation story: Flash vs GPT-4.1 routing,
  measure p95 and $/query.

**Honest caveat for DS roles:** a single GenAI project may not cover everything a
DS loop expects (experimentation, causal, forecasting). Pair this flagship with a
second, smaller project (e.g. an A/B-test or forecasting case study) to round it out.

## Questions they WILL ask — have crisp answers

- **"Why a cross-encoder and not just a better embedding model?"**
  Cross-encoders see query+doc *together* (full cross-attention), so they capture
  interactions bi-encoders can't. Too slow for first-stage retrieval, ideal for
  re-ranking a small candidate set.
- **"Manifest V3 constraint?"**
  Service workers are ephemeral — no long-running background process. So the agent
  and retrieval live in the FastAPI backend; the extension only extracts + calls HTTP.
- **"How are citations verified?"**
  Fact-check node computes a coverage signal (fraction of citations whose terms
  appear in the answer). Low coverage → warning + lower confidence. A stronger
  LLM-as-judge faithfulness check is on the roadmap.
- **"What would you do differently?"**
  (Have a real answer — e.g. "I'd move to pgvector sooner; the in-memory store
  rebuilds the index per session which won't scale," or "I'd add query rewriting
  before retrieval — it's a cheap win I deferred.")
