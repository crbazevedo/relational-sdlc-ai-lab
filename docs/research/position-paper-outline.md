# Position Paper Outline

Working title:

> Relational-Geometric Learning for SDLC and Software Agents

Core claim:

> SLMs and embeddings trained jointly over relational SDLC graphs can produce
> more reliable software engineering systems than SLMs with generic text RAG.

Contribution:

- not a single new loss;
- a unified architecture for relation-trained embeddings, SLM policy heads,
  graph retrieval, and verifiable SDLC feedback;
- public datasets and benchmarks.

Initial falsifiable hypothesis:

> Relation-trained embeddings improve file/test retrieval Recall@K and MRR over
> off-the-shelf embeddings on frozen public GitHub SDLC relation tasks.

Evidence status (2026-06-23):

- The initial hypothesis is **supported** for issue→fixing-PR: a relation-loss LoRA
  fine-tune beats the frozen embedder cross-repo, robustly (all 5 splits; within-split
  95% CIs exclude zero) and growing with data density (ΔR@1 +0.07 → +0.114). The
  contribution lives **in the representation**, not a post-hoc head.
- `diff→affected-test` is not yet won, but its blocker is now measured to be co-change
  **density** (an ingest artefact), not the method — the reachable ceiling rises from
  59.8% to 96.4% under real co-change history.
- Full ledger and rendered report: [research-roadmap.md](../research-roadmap.md) §3 and
  [docs/report/](../report/) (the dated program PDF).
