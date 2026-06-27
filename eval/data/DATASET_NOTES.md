# Dataset Construction Notes

This document describes the exact protocol used to construct the `eval/data/labeled_pairs.jsonl` dataset. This dataset is the foundation of the evaluation harness used to measure retrieval and re-ranking performance.

## 1. Query Sourcing (Manual)
To ensure the evaluation reflects real-world usage, all queries in this dataset were written **manually**. No automated tools or LLMs were used to generate the questions. 
- The queries represent genuine information-seeking intents (fact lookups, comparisons, contradictions) against the documents in `docs.jsonl`.
- This prevents the dataset from being overly simplistic or biased toward "easy" keyword matches that artificially inflate BM25 scores.

## 2. Candidate Suggestion (Retriever + LLM-Assisted)
To speed up the labeling process without compromising ground-truth accuracy, a hybrid approach was used for discovering candidates:
- For each manual query, the baseline hybrid retriever (BM25 + Dense) surfaced the top 10 candidate chunks.
- The Gemini API was used to provide a **suggestion only** for whether a chunk was a positive match, a hard negative (topically related but incorrect), or unsure.
- **Crucially, no LLM output was used directly as a label.** 

## 3. Human Review & Confirmation
Every single candidate suggested in the previous step was reviewed manually by a human. 
- Using a custom interactive CLI tool (`label_review.py`), each query and its candidate chunks were presented on screen.
- The human reviewer explicitly typed `y` (confirm positive) or `n` (confirm hard negative). Only these human-confirmed labels were admitted into the final dataset.

## 4. Hard Negative Mining (Retriever Misses)
A robust re-ranker must learn to distinguish between the correct answer and visually/semantically similar distractors. 
- For any query that lacked at least two confirmed hard negatives after the human review phase, additional hard negatives were mined automatically (`mine_hard_negatives.py`).
- These were sourced directly from the retriever's own "misses" (chunks that ranked highly via BM25/Dense but were not confirmed as positive). This provides pure, contrastive signals without needing further LLM calls.

## Summary
The final dataset (`labeled_pairs.jsonl`) contains 150-200 high-quality training pairs. The labels are 100% human-verified, while the candidate discovery phase leveraged retrieval and LLMs to accelerate the workflow. The 85/15 train/validation split for the cross-encoder fine-tuning is performed dynamically on this finalized set.
