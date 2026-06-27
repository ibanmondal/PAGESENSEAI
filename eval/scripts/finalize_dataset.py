"""Dataset finalization script.

Merges Phase 3 and Phase 4 output into the final format.
Output goes to eval/data/labeled_pairs.jsonl with the shape:
{"query": "...", "relevant_chunks": [...], "hard_negative_chunks": [...]}
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.core.logging import get_logger

log = get_logger("finalize_dataset")

def main():
    data_dir = ROOT / "eval" / "data"
    in_file = data_dir / "mined_negatives.jsonl"
    out_file = data_dir / "labeled_pairs.jsonl"

    if not in_file.exists():
        log.error(f"Missing {in_file}. Run mine_hard_negatives.py first.")
        return

    results = []
    with in_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            
            # Map into the final format
            final_obj = {
                "query": obj["query"],
                "relevant_chunks": [c["chunk_id"] for c in obj["confirmed_positives"]],
                "hard_negative_chunks": [c["chunk_id"] for c in obj["confirmed_hard_negatives"]]
            }
            results.append(final_obj)

    with out_file.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    log.info(f"Finalized dataset written to {out_file} with {len(results)} queries.")
    log.info("You are now ready to train the re-ranker!")

if __name__ == "__main__":
    main()
