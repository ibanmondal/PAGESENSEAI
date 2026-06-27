"""Suggest candidates for manual labeling using hybrid retrieval + LLM assistance.

Reads eval/data/raw_queries.txt and eval/data/docs.jsonl.
Outputs eval/data/candidates.jsonl containing top 10 chunks per query with optional LLM suggestions.
These are SUGGESTIONS ONLY and must not be used as final labels without human confirmation.
"""
import json
import sys
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.rag.chunking import chunk_document
from backend.app.rag.retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from backend.app.models.schemas import Chunk
from backend.app.agents.llm_client import build_router

log = get_logger("suggest_candidates")

def load_docs(docs_path: Path) -> List[Chunk]:
    chunks = []
    with docs_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # URL and title are arbitrary for eval chunking if missing
            doc_chunks = chunk_document(tab_id=obj["doc_id"], url=f"eval://{obj['doc_id']}", title=obj["doc_id"], text=obj["text"])
            chunks.extend(doc_chunks)
    return chunks

def get_llm_suggestions(router, query: str, top_chunks: List[Chunk]) -> Dict[str, str]:
    """Uses LLM to suggest which chunks are positive or hard negatives. Returns Dict[chunk_id, label]."""
    if not router.fast:
        return {}

    prompt = f"""You are an expert search relevance evaluator. 
Given the user query, review the following candidate search results.
For each result, classify it as:
- "positive" if it directly answers the query.
- "hard_negative" if it is topically related but does NOT answer the query, or answers a subtly different query (making it a good distractor for training).
- "unsure" otherwise.

Output ONLY a JSON object mapping the Result ID to the label string. Do not include any other text, markdown formatting, or explanations.

Query: "{query}"

Candidates:
"""
    for i, c in enumerate(top_chunks):
        prompt += f"\nResult ID: {c.chunk_id}\nContent: {c.text}\n"

    try:
        response = router.fast.complete(system="You are a relevance evaluator. Output raw JSON only.", user=prompt)
        # Strip potential markdown block
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        suggestions = json.loads(response.strip())
        return suggestions
    except Exception as e:
        log.warning(f"Failed to get LLM suggestions for query '{query}': {e}")
        return {}

def main():
    settings = get_settings()
    data_dir = ROOT / "eval" / "data"
    raw_queries_file = data_dir / "raw_queries.txt"
    docs_file = data_dir / "docs.jsonl"
    out_file = data_dir / "candidates.jsonl"

    if not raw_queries_file.exists():
        log.error(f"Missing {raw_queries_file}. Please create it and add queries.")
        return

    queries = []
    with raw_queries_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            q = line.strip()
            if q and not q.startswith("#"):
                queries.append(q)

    if not queries:
        log.error("No queries found in raw_queries.txt")
        return

    log.info("Loading documents and building corpus...")
    all_chunks = load_docs(docs_file)
    
    retriever = HybridRetriever(
        [BM25Retriever(), DenseRetriever(settings.embedding_model)],
        weights=[settings.bm25_weight, settings.dense_weight]
    )
    retriever.index(all_chunks)
    log.info(f"Indexed {len(all_chunks)} chunks.")

    router = build_router()

    results = []
    for query in queries:
        log.info(f"Processing query: {query}")
        hits = retriever.search(query, k=10)
        
        # Get LLM suggestions
        top_chunks = [h.chunk for h in hits]
        suggestions = get_llm_suggestions(router, query, top_chunks)

        candidates = []
        for i, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk.chunk_id
            label = suggestions.get(chunk_id, "unsure")
            if label not in ["positive", "hard_negative", "unsure"]:
                label = "unsure"

            candidates.append({
                "chunk_id": chunk_id,
                "text": hit.chunk.text,
                "retriever_rank": i,
                "llm_suggested_label": label,
                "doc_id": hit.chunk.tab_id
            })
        
        results.append({
            "query": query,
            "candidates": candidates
        })

    with out_file.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    log.info(f"Wrote {len(results)} queries with candidates to {out_file}")
    log.info("Run eval/scripts/label_review.py next to manually review these suggestions.")

if __name__ == "__main__":
    main()
