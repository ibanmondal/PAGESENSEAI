# Eval dataset

Two files power the retrieval eval harness:

- `docs.jsonl` — the document corpus. One JSON object per line: `{"doc_id": "...", "text": "..."}`.
- `evalset.jsonl` — the labeled queries. One JSON object per line:
  ```json
  {
    "query": "the user's natural-language question",
    "pool_doc_ids": ["doc_a", "doc_b", "doc_c"],
    "relevant_doc_ids": ["doc_a"],
    "notes": "optional — e.g. 'BM25 should dominate'"
  }
  ```

## How grading works

The harness grades at the **document level**, not chunk level: a retrieved chunk
counts as relevant iff its parent document is in `relevant_doc_ids`. This is the
standard protocol for RAG eval — the user cares whether the right *source* was
surfaced; re-ranking handles passage precision.

Each query's **pool** is the candidate corpus for *that query only* (mirrors how
a per-session RAG system works: small corpus per query). Retrievers index the
pool, return ranked chunks, the harness dedupes to docs and scores with
`nDCG@k`, `MRR`, `MAP`, `Recall@k`, `Precision@k`.

## Building a resume-grade set

The shipped 8-query set proves the harness runs. For real numbers, grow it to
**~150–200 queries** following this protocol:

1. **Collect realistic queries.** Browse docs/blogs you actually use; write down
   the real questions you had. Avoid keyword-stuffed queries (they make BM25 look
   artificially good). Include paraphrases and vague queries ("how does this work?").
2. **Build diverse pools.** For each query, pick 5–10 candidate docs including
   *hard negatives* — same-domain docs that are NOT the answer (e.g. for a BM25
   query, include a cross-encoder doc in the pool). Hard negatives are what
   separate good retrievers from lucky ones.
3. **Label honestly.** Mark only the doc(s) that genuinely answer the query. If
   two docs both answer, list both. When unsure, discard the query — bad labels
   make every metric meaningless.
4. **Hold out a test split.** Fine-tune the re-ranker on ~70% of queries, report
   final numbers on the held-out 30%. This is what makes the lift claim
   defensible in an interview.

Target a baseline spread that actually differentiates configs (e.g. BM25-only
nDCG@10 ~0.55, hybrid ~0.70, fine-tuned re-rank ~0.82). A set where every config
scores 1.0 tells you nothing — add harder negatives until the gaps open.
