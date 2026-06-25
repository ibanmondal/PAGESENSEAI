"""Fine-tune the cross-encoder re-ranker — the research centerpiece.

This turns "I used a pre-trained model" into "I fine-tuned a re-ranker and
measured the lift," which is the single most valuable sentence in a GenAI/ML
resume. The script:

  1. Loads labeled (query, doc, relevance) pairs from the eval dataset.
  2. Splits into train/val.
  3. Fine-tunes a cross-encoder on domain data using sentence-transformers.
  4. Saves weights to RERANKER_FINETUNED_PATH so the live pipeline uses them.
  5. The eval harness (eval/runner.py) then reports baseline vs fine-tuned.

Run:
    python -m eval.scripts.finetune_reranker \
        --data eval/data/evalset.jsonl --docs eval/data/docs.jsonl \
        --out eval/artifacts/reranker_finetuned

Then set RERANKER_FINETUNED_PATH=eval/artifacts/reranker_finetuned and re-run
the harness to get the headline before/after numbers.

Labeling strategy (honest, no leaks):
  - Positive pairs: query + chunks from a *relevant* doc.
  - Negative pairs: query + chunks from a *non-relevant* doc in the same pool.
  We mine negatives from the same query's pool (in-batch hard negatives), which
  is exactly the contrastive signal a re-ranker needs.
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

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.core.logging import get_logger   # noqa: E402
from backend.app.models.schemas import EvalRow    # noqa: E402
from backend.app.rag.chunking import chunk_document  # noqa: E402

log = get_logger("finetune")


def load_rows_and_docs(data_path: Path, docs_path: Path):
    rows: list[EvalRow] = []
    with data_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                o = json.loads(line)
                rows.append(EvalRow(
                    query=o["query"],
                    pool_doc_ids=o["pool_doc_ids"],
                    relevant_doc_ids=o["relevant_doc_ids"],
                ))
    docs: dict[str, str] = {}
    with docs_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                o = json.loads(line)
                docs[o["doc_id"]] = o["text"]
    return rows, docs


def build_pairs(rows: list[EvalRow], docs: dict[str, str], max_neg_per_query: int = 2):
    """Construct (query, passage, label) training pairs with hard negatives.

    Returns a list of [query, passage, label] where label is 1 (relevant) or 0.
    Negatives are sampled from the same pool (in-pool hard negatives).
    """
    pairs: list[list] = []
    rng = random.Random(42)
    for row in rows:
        relevant = set(row.relevant_doc_ids)
        # positives: every chunk from each relevant doc
        for doc_id in row.relevant_doc_ids:
            text = docs.get(doc_id, "")
            for ch in chunk_document(tab_id=doc_id, url="", title="", text=text):
                pairs.append([row.query, ch.text, 1])
        # negatives: chunks from non-relevant docs in the same pool
        neg_docs = [d for d in row.pool_doc_ids if d not in relevant]
        rng.shuffle(neg_docs)
        for doc_id in neg_docs[:max_neg_per_query]:
            text = docs.get(doc_id, "")
            chunks = chunk_document(tab_id=doc_id, url="", title="", text=text)
            if chunks:
                # take the first chunk as a hard negative
                pairs.append([row.query, chunks[0].text, 0])
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune cross-encoder re-ranker")
    parser.add_argument("--data", type=Path, default=Path("eval/data/evalset.jsonl"))
    parser.add_argument("--docs", type=Path, default=Path("eval/data/docs.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eval/artifacts/reranker_finetuned"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--base-model", default=None,
                        help="Override base model (default: settings.reranker_model)")
    args = parser.parse_args()

    settings = get_settings()
    base_model = args.base_model or settings.reranker_model

    rows, docs = load_rows_and_docs(args.data, args.docs)
    pairs = build_pairs(rows, docs)
    pos = sum(1 for p in pairs if p[2] == 1)
    neg = len(pairs) - pos
    log.info("Built %d pairs (%d pos / %d neg) from %d queries",
             len(pairs), pos, neg, len(rows))
    if len(pairs) < 20:
        log.warning("Very small training set (%d pairs). For real numbers, label "
                    "150-200 queries as described in eval/data/README.md.", len(pairs))

    # Shuffle then split (stratify-free is fine for this scale).
    rng = random.Random(42)
    rng.shuffle(pairs)
    split = max(1, int(len(pairs) * 0.85))
    train, val = pairs[:split], pairs[split:]
    log.info("Train: %d | Val: %d", len(train), len(val))

    try:
        from sentence_transformers import CrossEncoder
        from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
        from torch.utils.data import DataLoader
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "sentence-transformers + torch required for fine-tuning.\n"
            "pip install sentence-transformers torch"
        ) from e

    model = CrossEncoder(base_model, num_labels=1, max_length=512)
    log.info("Loaded base model: %s", base_model)

    train_loader = DataLoader(train, shuffle=True, batch_size=args.batch_size)
    evaluator = None
    if val:
        evaluator = CECorrelationEvaluator.from_input_examples(
            [(q, p, float(l)) for q, p, l in val], name="pagesense-dev"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_dataloader=train_loader,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        evaluator=evaluator,
        evaluation_steps=0,
        show_progress_bar=True,
        output_path=str(args.out),
    )
    model.save(str(args.out))
    log.info("Fine-tuned re-ranker saved -> %s", args.out)
    log.info("To use it: set RERANKER_FINETUNED_PATH=%s and re-run eval/runner.py",
             args.out)


if __name__ == "__main__":
    main()
