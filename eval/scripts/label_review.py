"""CLI interactive loop for human labeling of suggested candidates.

Reads eval/data/candidates.jsonl.
Shows one query and candidate at a time.
User types 'y' (positive), 'n' (hard negative), 's' (skip/unsure), 'q' (quit).
Output goes to eval/data/reviewed_candidates.jsonl.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.core.logging import get_logger

log = get_logger("label_review")

def main():
    data_dir = ROOT / "eval" / "data"
    in_file = data_dir / "candidates.jsonl"
    out_file = data_dir / "reviewed_candidates.jsonl"

    if not in_file.exists():
        log.error(f"Missing {in_file}. Run suggest_candidates.py first.")
        return

    # Load existing progress if any
    reviewed = {}
    if out_file.exists():
        with out_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                reviewed[obj["query"]] = obj

    queries_to_review = []
    with in_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            if obj["query"] not in reviewed:
                queries_to_review.append(obj)

    if not queries_to_review:
        print("All queries have been reviewed!")
        return

    print(f"Found {len(queries_to_review)} queries to review.")
    
    try:
        for q_obj in queries_to_review:
            query = q_obj["query"]
            candidates = q_obj["candidates"]
            
            print("\n" + "="*80)
            print(f"QUERY: {query}")
            print("="*80)

            confirmed_positives = []
            confirmed_hard_negatives = []

            for i, cand in enumerate(candidates, start=1):
                print(f"\n--- Candidate {i}/{len(candidates)} ---")
                print(f"LLM Suggestion: {cand['llm_suggested_label']}")
                print(f"Retriever Rank: {cand['retriever_rank']}")
                print(f"Document ID: {cand.get('doc_id', 'unknown')}")
                print("-" * 40)
                print(cand["text"])
                print("-" * 40)

                while True:
                    ans = input("Label: [y]es (positive), [n]o (hard negative), [s]kip, [q]uit: ").strip().lower()
                    if ans in ["y", "n", "s", "q"]:
                        break
                    print("Invalid input.")

                if ans == "q":
                    print("Quitting early. Progress saved so far.")
                    return
                elif ans == "y":
                    confirmed_positives.append(cand)
                elif ans == "n":
                    confirmed_hard_negatives.append(cand)

            reviewed[query] = {
                "query": query,
                "confirmed_positives": confirmed_positives,
                "confirmed_hard_negatives": confirmed_hard_negatives
            }

            # Save incrementally
            with out_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(reviewed[query]) + "\n")
                
    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved so far.")

    print("\nReview session complete. Next, run eval/scripts/mine_hard_negatives.py")

if __name__ == "__main__":
    main()
