"""Split evalset.jsonl into train.jsonl (70%) and test.jsonl (30%), deterministically.

This enforces the scientific discipline that makes a fine-tuning lift claim
defensible: the re-ranker is trained ONLY on train.jsonl and measured ONLY on
test.jsonl. No query appears in both. Run this once, then fine-tune on train
and eval on test.

Usage:
    python -m eval.scripts.split_dataset
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("eval/data/evalset.jsonl"))
    parser.add_argument("--train", type=Path, default=Path("eval/data/train.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("eval/data/test.jsonl"))
    parser.add_argument("--test-frac", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    with args.data.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_test = int(len(rows) * args.test_frac)
    test = rows[:n_test]
    train = rows[n_test:]

    args.train.write_text(
        "".join(json.dumps(r) + "\n" for r in train),
        encoding="utf-8",
    )
    args.test.write_text(
        "".join(json.dumps(r) + "\n" for r in test),
        encoding="utf-8",
    )

    print(f"Split {len(rows)} queries -> train={len(train)} / test={len(test)} (seed={args.seed})")


if __name__ == "__main__":
    main()
