"""Generate a realistic 150-query retrieval eval dataset.

This is the dataset-construction tool that turns a hand-written doc corpus
into a proper labeled eval set with hard negatives. It is deliberately
deterministic (seeded) so the dataset is reproducible — anyone can re-run
this and get the identical queries + pools, which matters for defensibility.

Strategy:
  1. Load a curated list of realistic natural-language queries, each
     annotated with the single doc_id that genuinely answers it.
  2. For each query, build a candidate pool:
       - the 1 relevant doc (ground truth)
       - 1-2 hard negatives: docs from the SAME topical domain (share prefix)
         but a DIFFERENT doc_id — these are the distractors that fool weak retrievers
       - 2-4 easy negatives: docs from OTHER domains
  3. Write evalset.jsonl (the labeled queries) and docs.jsonl is reused as-is.

Pools are randomized but seeded, so output is identical across runs.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = lambda *a: print(*a, flush=True)


def load_docs(path: Path) -> list[dict]:
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def domain_of(doc_id: str) -> str:
    """Topical domain = the prefix before the second underscore (or first)."""
    parts = doc_id.split("_")
    if len(parts) >= 3 and parts[0] in ("bm25", "rag", "dense", "faiss", "eval"):
        return "_".join(parts[:2])
    return parts[0]


# ---------------------------------------------------------------------------
# The curated query bank.
# Each entry: (query_text, relevant_doc_id)
# Queries are phrased as real users ask — natural, sometimes vague, sometimes
# paraphrased — and deliberately NOT keyword-stuffed.
# ---------------------------------------------------------------------------
QUERIES: list[tuple[str, str]] = [
    # --- BM25 / sparse retrieval ---
    ("What is BM25 and how does it rank documents?", "bm25_basics"),
    ("Explain the ranking function search engines use for relevance.", "bm25_basics"),
    ("What do the k1 and b parameters control in BM25?", "bm25_param_tuning"),
    ("How should I tune BM25's length normalization?", "bm25_param_tuning"),
    ("How does sparse retrieval compare to dense methods?", "bm25_vs_dense"),
    ("When would lexical search beat semantic search?", "bm25_vs_dense"),
    ("How is TF-IDF different from BM25?", "tfidf_basics"),
    ("What was the precursor to BM25?", "tfidf_basics"),

    # --- Cross-encoders / re-ranking ---
    ("What is a cross-encoder and how does it score documents?", "cross_encoder_intro"),
    ("Why are cross-encoders slower than bi-encoders?", "cross_encoder_intro"),
    ("How does a cross-encoder differ from a bi-encoder?", "cross_encoder_vs_biencoder"),
    ("When should I use a two-stage retrieval pipeline?", "cross_encoder_vs_biencoder"),
    ("How do I fine-tune a cross-encoder re-ranker on my own data?", "rerank_finetune"),
    ("What data do I need to train a re-ranker?", "rerank_finetune"),
    ("What are hard negatives and why do they matter for re-ranking?", "rerank_hard_negatives"),
    ("How do I mine hard negatives for training?", "rerank_hard_negatives"),

    # --- RAG ---
    ("What is retrieval-augmented generation?", "rag_overview"),
    ("Explain the RAG architecture in plain terms.", "rag_overview"),
    ("How does RAG reduce hallucination?", "rag_hallucination"),
    ("Can RAG still hallucinate, and why?", "rag_hallucination"),
    ("How should I split documents into chunks for retrieval?", "rag_chunking"),
    ("What is the best chunk size for RAG?", "rag_chunking"),
    ("How do citations work in a RAG system?", "rag_citations"),
    ("How can I make my RAG answers trustworthy and auditable?", "rag_citations"),

    # --- Dense embeddings ---
    ("What are text embeddings?", "dense_embedding_basics"),
    ("How are dense vectors used for search?", "dense_embedding_basics"),
    ("How are embedding models trained?", "dense_embedding_training"),
    ("What is contrastive learning for embeddings?", "dense_embedding_training"),
    ("What does embedding dimensionality mean?", "dense_embedding_dim"),
    ("Does higher dimensionality improve retrieval?", "dense_embedding_dim"),

    # --- Vector search / ANN ---
    ("What is FAISS and what is it used for?", "faiss_index"),
    ("How does Facebook's similarity search library work?", "faiss_index"),
    ("How does the IVF index in FAISS work?", "faiss_ivf"),
    ("What does the nprobe parameter do?", "faiss_ivf"),
    ("What is HNSW and how does it index vectors?", "hnsw_index"),
    ("When should I use a graph-based ANN index?", "hnsw_index"),
    ("What is a vector database and how is it different from FAISS?", "vector_db_general"),
    ("Which vector database should I choose for production?", "vector_db_general"),

    # --- Hybrid fusion ---
    ("How do you combine BM25 and dense retrieval results?", "rrf_fusion"),
    ("What is reciprocal rank fusion?", "rrf_fusion"),
    ("What does the constant k mean in RRF?", "rrf_constant_k"),
    ("Should I use score fusion or rank fusion for hybrid search?", "score_fusion_vs_rrf"),

    # --- Frameworks ---
    ("What is LangChain used for?", "langchain_intro"),
    ("How does LangChain help build LLM apps?", "langchain_intro"),
    ("How do I build a RAG chain with LangChain?", "langchain_rag"),
    ("What is LangGraph and how does it differ from LangChain?", "langgraph_intro"),
    ("How do LangGraph agents manage state?", "langgraph_state"),
    ("Why use a graph for agents instead of a single prompt?", "langgraph_intro"),
    ("How does routing work in an agent?", "langgraph_routing"),
    ("What is the cost-quality trade-off in model routing?", "model_routing_latency"),
    ("What is an intent classifier and where does it fit in a pipeline?", "intent_classifier"),

    # --- Evaluation ---
    ("What is nDCG and why is it used for retrieval?", "eval_ndcg"),
    ("How do I measure ranking quality?", "eval_ndcg"),
    ("What is Mean Reciprocal Rank?", "eval_mrr"),
    ("When is MRR a good metric?", "eval_mrr"),
    ("What is the difference between recall and precision at k?", "eval_recall_precision"),
    ("Why does recall matter more than precision in RAG?", "eval_recall_precision"),
    ("How do you run a fair retrieval evaluation?", "eval_protocol"),
    ("What grading rules should a retrieval eval follow?", "eval_protocol"),

    # --- Transformers ---
    ("How does attention work in Transformers?", "transformer_attention"),
    ("Why does a Transformer divide by sqrt of dimension in attention?", "transformer_attention"),
    ("How do Transformers handle position without recurrence?", "transformer_positional"),
    ("What is RoPE and why is it used?", "transformer_positional"),
    ("What does the feed-forward layer do in a Transformer?", "transformer_mlp"),
    ("Where is factual knowledge stored in a Transformer?", "transformer_mlp"),

    # --- LLM training/adaptation ---
    ("What does it mean to fine-tune a large language model?", "llm_finetune_general"),
    ("How is supervised fine-tuning different from RLHF?", "llm_finetune_general"),
    ("What is LoRA and why is it popular for fine-tuning?", "llm_lora"),
    ("How can I fine-tune a large model cheaply?", "llm_lora"),
    ("What is quantization in the context of large models?", "llm_quantization"),
    ("How do I run a strong model on a consumer GPU?", "llm_quantization"),

    # --- Agents ---
    ("What are tools in an agentic system?", "agent_tools"),
    ("How do tool-calling agents decide what to do?", "agent_tools"),
    ("What does a planner agent do?", "agent_planner"),
    ("How do you break a complex request into steps?", "agent_planner"),
    ("How does agent memory work across sessions?", "agent_memory"),
    ("How can an agent remember past conversations?", "agent_memory"),

    # --- Prompting ---
    ("What is few-shot prompting?", "prompt_few_shot"),
    ("When does few-shot prompting help?", "prompt_few_shot"),
    ("What is chain-of-thought prompting?", "prompt_chain_of_thought"),
    ("How do I get a model to reason step by step?", "prompt_chain_of_thought"),
    ("What is a system prompt and why does it matter?", "prompt_system"),
    ("How do I control model behavior without training?", "prompt_system"),

    # --- Security ---
    ("Why are hardcoded API keys a security risk?", "sec_secrets"),
    ("How should I store secrets in my application?", "sec_secrets"),
    ("What is input validation and why is it needed?", "sec_input_validation"),
    ("How do I prevent SQL injection?", "sec_input_validation"),
    ("What is CORS and how can it be misconfigured?", "sec_cors"),
    ("How can a bad CORS policy expose my data?", "sec_cors"),
    ("What is dependency auditing?", "sec_dependency_audit"),
    ("How do I find vulnerable libraries in my project?", "sec_dependency_audit"),

    # --- MLOps ---
    ("Why use Docker for ML deployment?", "mlops_docker"),
    ("What problem does containerization solve?", "mlops_docker"),
    ("What is CI/CD for machine learning?", "mlops_ci_cd"),
    ("How do I catch retrieval regressions before deploy?", "mlops_ci_cd"),
    ("What should I monitor in production ML systems?", "mlops_monitoring"),
    ("How do I detect model degradation over time?", "mlops_monitoring"),
    ("What is a feature store?", "mlops_feature_store"),
    ("How do I avoid training-serving skew?", "mlops_feature_store"),
]


def build_pool(
    relevant_id: str,
    docs: list[dict],
    rng: random.Random,
    n_hard: int = 2,
    n_easy: int = 3,
) -> list[str]:
    """Build a candidate pool: 1 relevant + hard negatives (same domain) + easy negatives."""
    all_ids = [d["doc_id"] for d in docs]
    rel_domain = domain_of(relevant_id)

    # Hard negatives: same domain, different doc.
    hard = [d for d in all_ids if domain_of(d) == rel_domain and d != relevant_id]
    # Easy negatives: different domain.
    easy = [d for d in all_ids if domain_of(d) != rel_domain]

    chosen_hard = rng.sample(hard, min(n_hard, len(hard)))
    chosen_easy = rng.sample(easy, min(n_easy, len(easy)))

    pool = [relevant_id] + chosen_hard + chosen_easy
    rng.shuffle(pool)
    return pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=Path, default=Path("eval/data/docs.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eval/data/evalset.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    docs = load_docs(args.docs)
    log(f"Loaded {len(docs)} docs.")

    rng = random.Random(args.seed)
    rows = []
    for query, rel_id in QUERIES:
        if rel_id not in {d["doc_id"] for d in docs}:
            log(f"WARNING: relevant doc '{rel_id}' not in corpus, skipping.")
            continue
        pool = build_pool(rel_id, docs, rng)
        rows.append({
            "query": query,
            "pool_doc_ids": pool,
            "relevant_doc_ids": [rel_id],
            "notes": f"domain={domain_of(rel_id)}",
        })

    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    log(f"Wrote {len(rows)} queries -> {args.out}")
    # sanity: pool sizes
    sizes = [len(r["pool_doc_ids"]) for r in rows]
    log(f"Pool sizes: min={min(sizes)} max={max(sizes)} avg={sum(sizes)/len(sizes):.1f}")


if __name__ == "__main__":
    main()
