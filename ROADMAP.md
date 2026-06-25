# Roadmap

This MVP is deliberately scoped. Shipping 4 polished things beats 20 broken ones.
Each item below is a *future milestone* — listed here to show scope awareness,
not to imply it's done. Order roughly = resume impact per unit effort.

## ✅ MVP (this repo)
- [x] Multi-tab + PDF ingestion with clean content extraction
- [x] Hybrid retrieval (BM25 + dense + Reciprocal Rank Fusion)
- [x] Cross-encoder re-ranking (base + fine-tuned)
- [x] Retrieval eval harness (nDCG / MRR / Recall@k / MAP) with unit-tested metrics
- [x] Re-ranker fine-tuning script with hard-negative mining
- [x] 3-node LangGraph agent (retrieve → generate → fact-check)
- [x] Model routing (fast vs strong)
- [x] Per-answer citations + confidence + warning
- [x] Chrome extension (Manifest V3): extraction + popup UI

## 🔜 High-impact next milestones
- [ ] **Expand eval set to ~150–200 realistic queries** with hard distractors. This is the single highest-leverage task: it turns "the harness works" into "here are real before/after numbers." See `eval/data/README.md`.
- [ ] **Run + document the fine-tuning lift** (baseline MAP/nDCG vs fine-tuned) — the headline resume number.
- [ ] **Query-rewriting** before retrieval (vague → specific). Cheap, big RAG win.
- [ ] **Latency budget doc**: p95 per stage with Flash vs GPT-4.1 routing.
- [ ] Persistent store (pgvector + Postgres) replacing the in-memory `SessionStore`.

## 🟡 Stretch (only after the above)
- [ ] Learned **intent classifier** to replace the rule-based router (A/B it against the current router using the same eval harness).
- [ ] YouTube transcript + GitHub repo ingestion (each is its own parser).
- [ ] React/TS sidebar UI replacing the vanilla popup.
- [ ] Multi-tab **Compare** mode UI (backend `mode=compare` already supports it).
- [ ] Citation faithfulness eval (LLM-as-judge: does each [#] actually support the claim?).
- [ ] Docker + CI (GitHub Actions) running the eval harness on every PR.

## ❌ Explicitly NOT doing (scope discipline)
- Browser automation / "agent controls my browser" — trust + ToS minefield.
- Knowledge graph visualization — looks cool, low ROI vs effort.
- Team workspaces / accounts — pure product work, not ML.
- Local on-device LLM in the extension — infeasible under MV3; would need a desktop backend.
