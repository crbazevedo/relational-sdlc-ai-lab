# Research Roadmap (living, experiment-driven)

This is the lab's research plan: paper-grounded, hypothesis-first, and flexible.
It is a **living document** — the experiment cards under
[`data/cards/examples/`](../data/cards/examples/) are the source of truth for
results; this roadmap holds the narrative, the open questions, and the decision
gates. Phase status (P0–P4) lives in [roadmap.md](roadmap.md); this file is the
deeper experimental layer beneath it.

---

## 1. Thesis and grounding in the position paper

> Embeddings (and later SLMs) trained over **verifiable SDLC relations** produce
> more reliable software-engineering systems than models that rely on generic text
> similarity.

We are in **MVP-1** of the paper (relational repository embeddings: retrieval,
localization, duplicate/relation prediction). What we have instantiated, and what
we have not:

| Paper element | Status | Where |
|---|---|---|
| Typed artifact/relation graph (§Formal Framework) | ✅ | `schemas/`, records/edges |
| Relation operator `h_v ≈ T_r(h_u)` | ✅ linear `M` | `relsdlc/tower.py:train_relation_map` |
| Bilinear `s_r(u,v)=hᵤᵀ W_r h_v` | ✅ diagonal `W_r`; ✅ low-rank `W_r=WqᵀWd` | `relsdlc/model.py`, `relsdlc/tower.py` |
| Topological margin loss `L_topo` | ✅ | margin triplet objective |
| Metrics: Recall@K, MRR, hard-neg, temporal/cross-repo generalization | ✅ | `relsdlc/metrics.py`, ablations |
| Pretrained-embedding substrate | ✅ frozen MiniLM | `data/pilot/embed_pilot.py` |
| `L_contrastive` (SupCon), `L_rel` (BCE), `L_graph` (spectral), `L_logic`, `L_align` | ⏳ partial/none | — |
| End-to-end fine-tuning (LoRA) | ❌ **not started** | Track A |
| Graph link prediction (GNN / KG-embedding) | ❌ | Track B |
| Relational SLM, GraphRAG, agent policy (MVP-2/3/4) | ❌ later | Track E |

---

## 2. Method invariants (the experimental protocol)

Every experiment obeys these, so results are trustworthy and comparable:

1. **Links are labels, not features.** Explicit references (`Fixes #N`, URLs, SHAs)
   are scrubbed from inputs ([`relsdlc/scrub.py`](../src/relsdlc/scrub.py)); a
   relation that can be regex-recovered is a label source, never a test.
2. **Frozen splits; cross-repo is the headline.** Held-out repositories measure
   generalization, not repo-specific memorization. Temporal splits where relevant.
3. **Pretrained embeddings are the substrate; the relational contribution is the
   delta over `embedder-cosine`.** A method only "wins" if it beats that control on
   the same split.
4. **Every run emits an experiment card**; exploratory results are labeled
   exploratory and never presented as release evidence.
5. **Reproducible from a clean checkout.** Heavy steps (embedding) are cached and
   committed; downstream runs on the cache with numpy alone, so CI needs no torch.
6. **Honest negatives are kept.** A result that refutes a hypothesis is a result.

---

## 3. Results ledger (what we know, with evidence)

| Exp | Hypothesis | Finding | Headline metric | Cards |
|---|---|---|---|---|
| Synthetic per-token | relation supervision > vanilla/IDF when the link is per-token | **confirmed** | R@1 0.82 vs IDF 0.38 vs vanilla 0.11 | `synth-*` |
| Cross-token synthetic | only a cross-token operator can bridge disjoint vocab | **confirmed** | tower R@1 0.83 vs all-else 0.01 | — |
| Real issue→PR (explicit link) | does the synthetic win transfer? | **no** — surface-rich; IDF ties diagonal | IDF R@1 0.46 | `gh-pilot-*` |
| Real, de-referenced, cross-repo (bag-of-tokens) | learned head generalizes cross-repo? | **no** — tower 0.24 < vanilla 0.39 | — | `gh-xrepo-*` |
| Embeddings, cross-repo | pretrained embeddings generalize where tokens don't? | **yes** — embedder-cosine wins | R@1 0.59 vs IDF 0.46 | `gh-embed-cosine` |
| Relation head on frozen embeddings | does our operator add value on top? | **no** at pilot scale — from-scratch overfits (0.19), identity-init ties (0.59) | — | `gh-embed-tower`, `gh-embed-relmap` |

**Synthesis:** embeddings settle the *substrate* question; a bolt-on operator on
*frozen* vectors has no headroom at pilot scale. The relational contribution must
therefore live **inside the representation** (fine-tuning) or **in graph
structure** (link prediction) — not as a post-hoc head.

---

## 4. Open questions (falsifiable)

- **Q1 (core).** Does fine-tuning a small embedder with the relation/contrastive
  loss (LoRA) beat frozen `embedder-cosine` cross-repo? *This is where "our
  contribution" can reshape the representation; findings above say a frozen-head
  cannot.*
- **Q2.** Does link prediction over the typed SDLC graph (GNN / KG-embedding) add
  signal beyond pairwise text cosine?
- **Q3.** Do relations where surface text is weaker (`diff→affected-test`,
  `log→file`) show a larger relational lift than `issue→fixing-PR`?
- **Q4.** Does cross-repo `same-bug-class` retrieval benefit from relation-trained
  embeddings (the latent, non-hyperlinked relation)?
- **Q5.** Does scale (more repos / pairs) give a learned head the headroom it
  lacked at pilot scale?
- **Q6.** Code-specific embedder (e.g. a code embedding model) vs general
  (MiniLM) — how much does the base matter?

---

## 5. Tracks (parallelizable, each with a decision gate)

### Track A — Representation: fine-tune with the relation loss  *(NEXT)*
- **Paper:** `L_contrastive` + `L_topo` reshaping `E_θ` (the embedder).
- **Prereq:** `[embed]` extra (transformers+torch); a small LoRA/last-layer train loop.
- **Experiment:** LoRA-tune MiniLM (and/or a code embedder) on **train-repo** `fixes`
  pairs with InfoNCE + hard negatives; cache the tuned embeddings; eval cross-repo
  on the committed split. Control = frozen `embedder-cosine`.
- **Success:** tuned beats frozen `embedder-cosine` R@1 (currently 0.59) by a
  margin that survives a held-out-repo re-split.
- **Gate:** win → add auxiliary relation heads + scale (Track D). No win → diagnose
  (capacity / data / loss / base model via Q6) before scaling.

### Track B — Graph: link prediction over the SDLC graph
- **Paper:** `L_graph` (spectral) + KG-embedding relation scoring; GraphRAG seed.
- **Prereq:** **graph enrichment** — extend the pilot ingest to fetch PR changed
  files (issues↔PRs↔files↔commits↔tests), yielding `modifies`/co-change edges.
- **Experiment:** inductive GNN (GraphSAGE/R-GCN) on **pretrained node features**
  (so it transfers to unseen repos), multi-relation link prediction; eval
  `issue→PR` and `diff→test` cross-repo. Control = `embedder-cosine` + Track A.
- **Success:** beats the best non-graph system on a relation cosine can't capture.
- **Gate:** win → richer graph + KG operators (RotatE/ComplEx) for asymmetric
  relations. No win → graph structure isn't load-bearing at this scale; revisit at
  Track D scale.

### Track C — Tasks & labels
- **C1 `diff→affected-test`, `log→file`** (needs Track B's file edges): test Q3.
- **C2 cross-repo `same-bug-class`** (Q4): tagging via **BERTopic + LLM mapped to a
  curated SWE bug-class ontology** (concurrency, DB synchronization, networking,
  memory/CPU-bound, …). Labels must be independent of the body text (avoid
  circularity); cross-repo split + textually-near hard negatives.
- **Gate:** each task is non-degenerate (not regex-recoverable) before it counts.

### Track D — Scale (only on signal)
- Tier-2 (200–500 repos) once a method shows signal at pilot scale; Tier-3 only if
  scaling evidence is needed. Do not scale before signal.

### Track E — Beyond MVP-1 (later)
- Relational SLM (MVP-2): QLoRA SLM with relation/policy heads for review, test
  suggestion, risk. GraphRAG subgraph packer. Agent policy + outcome learning.
  Explicitly deferred until MVP-1 shows a relational win.

---

## 6. Sequencing & decision policy

```
NOW ─► Track A (LoRA fine-tune)         ── highest leverage; tests Q1, the core claim
  ║                                         gate: beats embedder-cosine cross-repo?
  ╠═► Track B prereq (graph enrichment) ── independent; unblocks GNN + diff→test/log→file
  ║
refine ─► update §3 ledger + gates each wave ─► branch per gate outcome
  ║
scale (Track D) ONLY when a method shows signal ─► then Track E (SLM/agent)
```

- One change at a time vs the `embedder-cosine` control, so every delta is attributable.
- Gates are decision points, not milestones — a "no" reroutes (diagnose / change
  base model / scale), it does not stall the program.

---

## 7. Compute discipline

CPU-first; small models (MiniLM / a small code embedder; a small SLM later);
**LoRA, not full fine-tuning**; cache embeddings and commit them so experiments
replay cheaply; reach for GPU/scale only when a method has shown signal worth
scaling. The benchmark stays small until the signal says "feed me."

---

## 8. How to add an experiment

1. State a falsifiable hypothesis (add to §4) and the control it must beat.
2. Use the method invariants (§2): de-referenced inputs, frozen cross-repo split.
3. Run it; emit an experiment card per system (`data/cards/examples/*.experiment-card.json`).
4. Record the outcome in the §3 ledger; let the relevant gate (§5) decide the next branch.
