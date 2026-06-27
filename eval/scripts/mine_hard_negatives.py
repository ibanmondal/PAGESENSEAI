"""Mine additional hard negatives from retriever misses.

For any query that ends Phase 3 with fewer than 2 confirmed hard negatives,
this script pulls additional hard negatives directly from the retriever's
own top-10 misses (chunks ranked highly by BM25/dense but not confirmed positive).
Output goes to eval/data/mined_negatives.jsonl.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.rag.chunking import chunk_document
from backend.app.rag.retrieval import BM25Retriever, DenseRetriever, HybridRetriever

log = get_logger("mine_hard_negatives")

def load_docs(docs_path: Path):
    chunks = []
    with docs_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_chunks = chunk_document(tab_id=obj["doc_id"], url=f"eval://{obj['doc_id']}", title=obj["doc_id"], text=obj["text"])
            chunks.extend(doc_chunks)
    return chunks

def main():
    settings = get_settings()
    data_dir = ROOT / "eval" / "data"
    in_file = data_dir / "reviewed_candidates.jsonl"
    docs_file = data_dir / "docs.jsonl"
    out_file = data_dir / "mined_negatives.jsonl"

    if not in_file.exists():
        log.error(f"Missing {in_file}. Run label_review.py first.")
        return

    queries_data = []
    with in_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            queries_data.append(json.loads(line))

    # Check if we need to retrieve
    needs_mining = [q for q in queries_data if len(q["confirmed_hard_negatives"]) < 2]
    if not needs_mining:
        log.info("All queries already have at least 2 confirmed hard negatives. No mining needed.")
        # Just copy input to output
        with out_file.open("w", encoding="utf-8") as fh:
            for q in queries_data:
                fh.write(json.dumps(q) + "\n")
        return

    log.info("Loading documents and building corpus for mining...")
    all_chunks = load_docs(docs_file)
    
    retriever = HybridRetriever(
        [BM25Retriever(), DenseRetriever(settings.embedding_model)],
        weights=[settings.bm25_weight, settings.dense_weight]
    )
    retriever.index(all_chunks)

    results = []
    for q_data in queries_data:
        query = q_data["query"]
        positives = {c["chunk_id"] for c in q_data["confirmed_positives"]}
        negatives = {c["chunk_id"] for c in q_data["confirmed_hard_negatives"]}
        
        while len(q_data["confirmed_hard_negatives"]) < 2:
            hits = retriever.search(query, k=20)
            added = False
            for hit in hits:
                cid = hit.chunk.chunk_id
                if cid not in positives and cid not in negatives:
                    q_data["confirmed_hard_negatives"].append({
                        "chunk_id": cid,
                        "text": hit.chunk.text,
                        "retriever_rank": len(q_data["confirmed_hard_negatives"]) + 1,
                        "llm_suggested_label": "mined_retriever_miss",
                        "doc_id": hit.chunk.tab_id
                    })
                    negatives.add(cid)
                    added = True
                    break # Break inner loop, check condition again
            if not added:
                log.warning(f"Could not find enough hard negatives for query: '{query}'")
                break # Give up on this query if no more chunks

        results.append(q_data)

    with out_file.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
            
    log.info(f"Mined hard negatives for {len(needs_mining)} queries. Output saved to {out_file}")
    log.info("Run eval/scripts/finalize_dataset.py next.")

if __name__ == "__main__":
    main()
