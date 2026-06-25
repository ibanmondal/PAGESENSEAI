# PageSense AI — Multi-Tab RAG Copilot

> A browser copilot that reads your open tabs + PDFs and answers with citations, powered by a **hybrid retrieval pipeline (BM25 + dense + cross-encoder re-ranker)** with a **fine-tuned re-ranker** and a measurable eval harness.

Most browser-AI extensions answer questions about *one* page. PageSense indexes **everything you have open**, retrieves across all of it, and grounds every answer in cited sources with a confidence score.

---

## Why this project exists

The job market rewards a specific story: *"I built a RAG system, then fine-tuned the re-ranker and lifted retrieval quality by measurable points on a labeled eval set I built myself."* This repo is engineered to make that sentence true and defensible. Every design choice favors **measurability over feature count**.

| Capability | Where it lives |
|---|---|
| Hybrid retrieval (BM25 + dense + RRF fusion) | `backend/app/rag/retrieval.py` |
| Cross-encoder re-ranking | `backend/app/rag/reranker.py` |
| Re-ranker fine-tuning | `eval/scripts/finetune_reranker.py` |
| Retrieval eval harness (nDCG / MRR / Recall@k) | `eval/runner.py`, `eval/metrics.py` |
| Multi-step LangGraph agent (retrieve→generate→fact-check) | `backend/app/agents/graph.py` |
| Model routing (fast vs strong) | `backend/app/agents/llm_client.py` |
| Chrome extension (Manifest V3) | `extension/` |

---

## Architecture

```
Browser Extension (MV3)
  content script ──extract clean text──►  background service worker
                                              │ POST /ingest, /query
                                              ▼
FastAPI backend
  routes ─► SessionStore ─► HybridRetriever ─► CrossEncoderReRanker
                                              │ top-k cited chunks
                                              ▼
                          LangGraph: retrieve → generate → fact-check
                                              │
                          ModelRouter: Flash (cheap) vs GPT-4.1 (hard)
                                              ▼
                              Answer + citations + confidence
```

The agent and all retrieval work run **server-side** — the extension is a thin extraction + HTTP layer. This is required by Manifest V3 (service workers are ephemeral) and keeps the heavy ML out of the browser.

---

## Quick start

### 1. Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example ../.env           # then add your API keys
cd .. && uvicorn backend.app.main:app --reload --port 8000
```

### 2. Verify the retrieval core (no LLM keys needed)
```bash
python -m pytest eval/test_metrics.py -v          # 11 metric unit tests
python -m eval.runner --config bm25               # BM25 baseline
python -m eval.runner --config all                # bm25 | dense | hybrid | hybrid+rerank
```

### 3. Extension
1. Open `chrome://extensions`
2. Enable **Developer mode** → **Load unpacked** → select the `extension/` folder
3. Open some docs/tabs → click the PageSense icon → **Index all tabs** → ask a question

---

## The eval harness (resume centerpiece)

This is the part that turns "I used an API" into "I'm an ML engineer."

```bash
# 1. Measure baselines (4 configs over the labeled set)
python -m eval.runner --config all

# 2. Fine-tune the re-ranker on your domain labels
python -m eval.scripts.finetune_reranker \
    --out eval/artifacts/reranker_finetuned

# 3. Point the pipeline at your fine-tuned weights and re-measure
RERANKER_FINETUNED_PATH=eval/artifacts/reranker_finetuned \
    python -m eval.runner --config hybrid+rerank
```

**Metrics** (doc-graded binary relevance): `MRR`, `MAP`, `nDCG@k`, `Recall@k`, `Precision@k` — implemented as pure functions in `eval/metrics.py`, covered by 11 unit tests so the numbers are provably correct.

**Seed dataset** (`eval/data/`): 8 queries × 8 docs. This proves the harness works; for resume-grade numbers you expand it to ~150–200 realistic, hand-labeled queries (see `eval/data/README.md`).

### Current results (seed set — intentionally easy, all configs near-ceiling)

| config | MRR | nDCG@10 | MAP |
|---|---|---|---|
| bm25 | 1.000 | 0.990 | 0.979 |
| dense | 1.000 | 0.990 | 0.979 |
| hybrid | 1.000 | 0.990 | 0.979 |
| **hybrid+rerank** | **1.000** | **1.000** | **1.000** |

The re-ranker already shows lift (MAP 0.979 → 1.00) even on an easy set. On a realistic set with hard distractors the gaps open up — that's the story to tell.

---

## How to angle this for different roles

- **GenAI / AI Engineer:** lead with the agent + hybrid retrieval + routing + fine-tuning.
- **ML Engineer:** lead with the learning-to-rank framing — labeled eval set, controlled A/B of retrievers, before/after metrics.
- **Data Scientist:** lead with the evaluation science — metric choice, grading protocol, train/val split, statistical significance of the re-ranker lift.

See [`docs/INTERVIEW.md`](docs/INTERVIEW.md) for the exact talking points.

---

## Roadmap

What's **in** this MVP vs deliberately **deferred** — scope discipline is itself a signal. See [`ROADMAP.md`](ROADMAP.md).

Shipped: multi-tab ingest, hybrid retrieval, cross-encoder re-rank, fine-tuning script, eval harness, 3-node agent, model routing, citations + confidence, working MV3 extension.

---

## License

MIT
