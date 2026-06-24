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

<a name="audit"></a>
### Audit (R13)

An independent adversarial audit stress-tested the five headline claims:
**verdict — sound, zero CRITICAL issues.** Verified mechanically: cross-repo splits
genuinely disjoint (0 train-repo candidates in any test query); reference scrub
leaks no gold-pair numbers; train/eval separation (182 train pairs, all train-repo);
the graph leakage guard is load-bearing (gold edge unreachable post-guard); the
5-split study uses 5 distinct partitions; **every numpy result reproduces the
committed cards byte-for-byte**. Responses landed: a CI provenance test
(`tests/test_provenance.py`) now asserts the disjointness + train-pair count without
torch (closing the snapshot-trust gap); R@5/R@10 are footnoted as near-ceiling at a
13-candidate pool (R@1/MRR are the discriminating metrics).

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
| **LoRA fine-tune (Track A)** | does the relation loss *inside* the rep beat the frozen embedder cross-repo? | **YES** (pilot-scale) — R@1 0.59→0.66, MRR 0.73→0.79; a head on tuned vecs still adds nothing | R@1 0.66 vs frozen 0.59 | `gh-finetune-*` |
| **Multi-split confidence (Track D)** | is the LoRA win robust, not split luck? | **YES** — positive on all 5 held-out-repo splits | ΔR@1 +0.061±0.021, ΔMRR +0.052±0.010 | `gh-scale-*` |
| **Graph probe (Track B)** | does typed-graph aggregation add signal beyond cosine? | **small but real on issue→PR; stacks with LoRA**; diff→test needs denser graph | LoRA+graph R@1 0.69 vs LoRA 0.66 | `gh-gnn-*` |
| **Q6 base model (R12B)** | does a code/stronger base help the frozen substrate? | **embedding-tuned matters, not "code"** — bge edges past MiniLM; code-MLM collapses | bge 0.598 / MiniLM 0.592 / codebert 0.144 | `gh-code-*` |
| **Learned R-GCN (R12B)** | does a learned GNN beat parameter-free aggregation? | **no at pilot scale** — learned head overfits ~180 pairs | rgcn 0.575 < free-agg 0.609 | `gh-rgcn-*` |
| **Scale to ~55 repos (R12A)** | does the bag-of-tokens finding hold at larger scale? | **yes** — IDF still best, diagonal still ties | IDF 0.444 ≥ vanilla 0.333 | `gh-scale2-*` |
| **Relational SLM v0 (R12C)** | can a relation-packed subgraph drive an SLM (MVP-2)? | **dry-run runs** — fixing PR top-5 18/20; small SLM grounds 2/3 (no benchmark claim yet) | retrieval ≈0.9 top-5 | `slm-outputs/` |
| **LoRA-at-scale (R13A)** | does the LoRA win hold on 55 repos? | **yes — and grows** (14 held-out test repos) | ΔR@1 +0.080, ΔMRR +0.072 | `gh-scale-lora-*` |
| **Code-embedding base (R13B)** | does a code base beat the general substrate? | **no** — axis is embedding-tuned, not "code" (monotone in tuning) | codebert 0.14 < unixcoder 0.45 < st-codesearch 0.55 < MiniLM 0.59 ≈ bge 0.60 | `gh-code2-*` |
| **Full text (R14)** | does de-truncating bodies (500→8000) help? | **no — it HURTS** (paired control, truncation the only variable) | every system −0.09 to −0.15 R@1; embedder 0.69→0.55 | `gh-full-*` |
| **Chunked MaxP (R15)** | does MaxP over chunks beat FirstP for issue→PR? | **no** — signal is front-loaded; FirstP wins every chunk size; SumP collapses (length bias) | FirstP@512 0.701 > MaxP 0.668; queued for deep-signal tasks | `gh-chunk-*` |
| **Code-embedding base, pinned (R15B)** | does a true code+embedding-tuned base win? | **qualified yes** — jina-code (transformers<5) best R@5 ever, ties MRR, loses R@1 by a hair | R@5 0.960 (best); R@1 0.580 vs MiniLM 0.592 / bge 0.598 | `gh-code3-*` |
| **Deep-content chunking (R16A)** | does MaxP beat FirstP where signal is deep (diff→affected-test)? | **yes** — the mirror of R15; MaxP wins at every chunk size, biggest win at small chunks | ΔR@1 +0.346 / +0.171 / +0.112 (chunks 256/512/1024) | `gh-content-*` |
| **Dense Tier-2 baseline (R16B)** | does the bag-of-tokens finding hold at 78 repos with ~3.5× denser per-repo coverage? | **yes** — IDF still beats vanilla cross-repo; density did not erase the gap | IDF R@1 0.389 vs vanilla 0.287 (+0.102) | `gh-tier2-{vanilla,idf}-*` |
| **LoRA-at-Tier-2 (R16C)** | does the LoRA win hold at dense ~80 repos? | **yes — and grows further** (32 held-out test repos, density ~35 q/repo) | ΔR@1 +0.114 (0.515→0.629), ΔMRR +0.101 | `gh-tier2-lora-*` |
| **LoRA sweep (R16D)** | is +0.114 a floor or near-tuned (rank × harder negatives)? | **near-tuned** — rank saturated (r8≈r16≈r32, ±0.003); harder in-batch negatives are the only lever and the next gain wants GPU memory | best r16-b48 ΔR@1 +0.126 (vs +0.114); rank flat | `tier2-sweep-results.json` |
| **Graph-lift sweep (R16E)** | is the R11B graph lift a tuned knife-edge or robust; can multi-hop rescue diff→test? | **robust plateau + structure-bound** — issue→PR lift positive across α∈[0,0.75], hops=1≡hops=2 (R11B point sits on a plateau, 1 hop suffices); diff→test flat at every (α,hops) because 46.9% of gold tests are isolated after the leakage guard | issue→PR LoRA+graph 0.690 (h1, α0.25); diff→test reachable ceiling 59.8% | `gh-graphsweep-*` |
| **LoRA-win CIs (R17a)** | does the headline LoRA delta survive a within-split CI, and where does it come from? | **yes — both query- and repo-cluster 95% CIs exclude zero; broad but not uniform** (5/8 repos improve, 2 regress slightly; net +11 rank-1 flips, sign-test p≈0.043) | ΔR@1 +0.063, CI [+0.006,+0.121] (repo-cluster [+0.007,+0.122]); ΔMRR +0.064, CI [+0.027,+0.102] | `bootstrap-ci-results.json` |
| **diff→test density (R17b)** | is the 59.8% diff→test ceiling method-bound or an ingest-depth (density) artefact? | **density artefact, confirmed** — gold test files are heavily co-changed in real history (median 35 commits each); only 12/110 touched by ≤1 change | reachable ceiling 59.8% → **96.4%** (isolation 46.9% → 4.4%) under real co-change | `diff2test-density-results.json` |
| **Negatives lever (R18)** | does pushing R16D's negative-pool lever (more vs harder) grow the Tier-2 LoRA win, now that the M5 GPU lifts the CPU memory cap? | **harder, not more** — random pool flat (b32/48/96 +0.12–0.13; b192/384 OOM the M5), but same-repo **hard** batching at matched pool is the new best; Tier-2 CI excludes zero decisively (31/32 repos up) | repo-hard ΔR@1 **+0.137** (vs matched random +0.120; R16D best +0.126), CI [+0.112,+0.164] (repo-cluster [+0.107,+0.173]) | `negatives-sweep-results.json` |
| **Hardness multi-seed (R19)** | is R18's hardness gain real across seeds or MPS noise? | **real** — repo-hard beats matched random on **both** seeds with a tight paired gap (mean +0.020, std ~0.001), above the ~±0.008 MPS noise; seed-0 reproduces R18's +0.1375 exactly (the +0.146 first-run was a memory-leak-state artefact) | random mean +0.122 vs **repo-hard mean +0.142** (paired gap +0.019/+0.021); mining (H2) deferred | `hardneg-confirm-results.json` |

**Synthesis (R3→R12, refined by R13/R14).** The relational win comes from the **base representation**:
an embedding-tuned model as substrate, **LoRA reshaping it with the relation loss**
(robust: positive on all 5 cross-repo splits), with a **thin parameter-free graph
lift** on top (LoRA+graph R@1 0.69). Everything *bolted on frozen vectors* — a
from-scratch tower, an identity-init operator, a learned R-GCN — **overfits at pilot
scale and does not help.** The base finding (IDF ≫ vanilla; diagonal ties IDF) holds
at 55 repos. A relation-packed subgraph already drives a small SLM (dry-run). The
single lever that consistently pays off is *changing the representation itself*;
the single thing that consistently fails is *adding a learned head over frozen
features*.

R13/R14 sharpened two beliefs: (a) the base model matters along the
**embedding-tuned** axis, not "code-pretrained" — a code base does not beat a strong
general embedder at pilot scale; (b) **more text is not better** — a paired control
shows de-truncating bodies (500→8000 chars, 256→512 tokens) *hurts* every system by
0.09–0.15 R@1, because the first ~500 chars (title + lede) carry the signal and the
rest dilutes both lexical and mean-pooled-embedding methods. The next real unlocks
are therefore **scale (more repos), a code-*embedding* base via a pinned
transformers<5 env, smarter use of long text (salient-section / better pooling), and
a trained relational SLM** — not naive "use the whole body."

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

### Track A — Representation: fine-tune with the relation loss  *(DONE, pilot — WON)*
- **Paper:** `L_contrastive` + `L_topo` reshaping `E_θ` (the embedder).
- **Result:** LoRA (r=8, 0.48% params) on MiniLM, InfoNCE on 182 train-repo pairs,
  beat frozen `embedder-cosine` cross-repo (R@1 0.59→0.66, MRR 0.73→0.79). See
  [ablation-finetune.md](ablation-finetune.md).
- **Gate fired: WIN.** Next per the gate → **scale (Track D)** with multi-split CIs,
  and **code-specific base** (Q6). A head on tuned vectors still adds nothing — keep
  the contribution in the representation.
- **CIs landed (R17a):** the re-split/CI open item is now closed on the default split —
  both a query bootstrap and a **repo-cluster** bootstrap put ΔR@1's 95% CI above zero
  ([+0.006,+0.121]) and ΔMRR's more decisively ([+0.027,+0.102]); the per-repo
  decomposition shows the win is **broad but not uniform** (5/8 repos improve, 2 regress
  slightly). Together with R11A (all-5-splits) the win is robust to which repos are
  held out, to query resampling, and to repo correlation. Still open: **multiple seeds**,
  and a CI on the larger Tier-2 delta (needs torch to regenerate the gitignored caches).
  [ablation-bootstrap-ci.md](ablation-bootstrap-ci.md)

### Track B — Graph: link prediction over the SDLC graph  *(prereq DONE; probe DONE)*
- **Paper:** `L_graph` (spectral) + KG-embedding relation scoring; GraphRAG seed.
- **Prereq DONE (R10B):** graph enriched with 1,356 `modifies` edges + 497 file /
  239 test nodes ([graph-enrichment.md](graph-enrichment.md)).
- **Probe DONE (R11B):** a training-free typed mean-aggregation
  ([ablation-gnn.md](ablation-gnn.md)) adds a **small real lift on issue→PR** and
  **stacks with LoRA** (LoRA+graph R@1 0.69 vs LoRA 0.66). `diff→test` fails at
  pilot sparsity (≈47% of relevant tests isolated once the gold edge is removed).
- **Learned-GNN tried (R12B):** a learned 1-hop R-GCN (init at frozen, InfoNCE)
  **did NOT beat** the parameter-free aggregation at pilot scale (0.575 < 0.609) —
  it overfits ([ablation-rgcn.md](ablation-rgcn.md)). So free aggregation stays the
  best graph method here; a learned GNN is re-gated on **Track-D scale** (more
  supervision) + a richer/regularized multi-hop R-GCN. `diff→test` still needs
  denser co-change.
- **Robustness sweep DONE (R16E):** the free-aggregation lift was measured at one
  point `(α=0.5, hops=2)`; an `α×hops` sweep ([ablation-graph-sweep.md](ablation-graph-sweep.md))
  shows the issue→PR lift is a **plateau** (positive across α∈[0,0.75], both feature
  sets) and **saturates at 1 hop** (hops=1≡hops=2) — so 0.690 is robust and cheap,
  not a tuned spike. The learned-GNN bar is thus ~0.690 (LoRA)/~0.621 (frozen), not
  raw cosine. The sweep also **quantifies the `diff→test` blocker**: 46.9% of gold
  test nodes are isolated (degree-0) after the leakage guard → a 59.8% reachable
  ceiling, feature- and hop-independent. Confirms the limiter is co-change **density
  (a Track-D data problem)**, not the aggregation method.
- **Density confirmed against ground truth (R17b):** a live co-change probe
  ([ablation-diff2test-density.md](ablation-diff2test-density.md)) shows the gold test
  files are in fact heavily co-changed (**median 35 commits each**; only 12/110 touched
  by ≤1 change). The pilot's 46.9% isolation was an **ingest-depth artefact**: with real
  history the reachable ceiling rises **59.8% → 96.4%**. So the structural blocker is
  removable by denser ingest — the diff→test retrieval re-eval (which needs torch to
  embed the new PR nodes) is now a worthwhile **Track-D/GPU** follow-up, no longer
  capped below 60% by construction.

### Track C — Tasks & labels
- **C1 `diff→affected-test`, `log→file`** (needs Track B's file edges): test Q3.
- **C2 cross-repo `same-bug-class`** (Q4): tagging via **BERTopic + LLM mapped to a
  curated SWE bug-class ontology** (concurrency, DB synchronization, networking,
  memory/CPU-bound, …). Labels must be independent of the body text (avoid
  circularity); cross-repo split + textually-near hard negatives.
- **Gate:** each task is non-degenerate (not regex-recoverable) before it counts.

### Track D — Scale (only on signal)  *(confidence DONE; repo-scale DENSE)*
- **Multi-split confidence DONE (R11A):** the LoRA win is positive on **all 5**
  held-out-repo splits — ΔR@1 +0.061±0.021 ([ablation-scale.md](ablation-scale.md)).
- **Repo-scale DONE (R12A → R13A):** a ~55-repo dataset ([scale-dataset.md](scale-dataset.md));
  the bag-of-tokens finding holds (IDF 0.444 ≥ vanilla 0.333) and the LoRA win
  **grows** (ΔR@1 +0.080).
- **Dense Tier-2 DONE (R16B/C):** 78 repos at **~35 queries/repo** (~3.5× denser),
  2,744 benchmark queries ([tier2-dataset.md](tier2-dataset.md)). The bag-of-tokens
  finding holds (IDF R@1 0.389 ≥ vanilla 0.287); the LoRA delta **grows with
  density** to ΔR@1 **+0.114** (0.515→0.629), the largest in the program (pilot +0.07
  → scale +0.080 → Tier-2 +0.114).
- **LoRA sweep DONE (R16D):** +0.114 is **near-tuned, not a floor**
  ([ablation-lora-sweep.md](ablation-lora-sweep.md)). Rank is saturated (r8≈r16≈r32,
  ±0.003 R@1); the only lever that moves is **harder/more in-batch negatives** —
  pool 32→48 gives the best recipe yet (r16-b48, ΔR@1 **+0.126**). Larger negative
  pools are capped by CPU memory, so the next gain is a GPU lever. **Open:** Tier-2
  full breadth (200–500 repos) is a GPU-scale follow-up.
- **Negatives lever resolved (R18):** the first wave trained on the **Apple M5 GPU
  (MPS)** ([ablation-negatives-sweep.md](ablation-negatives-sweep.md)) settled R16D's
  "harder/more negatives" into **harder, not more**. More random negatives is flat
  (b32/48/96 +0.12–0.13) and the largest pools (b192/384) **OOM the M5** (pool ceiling
  ≈ b96) — so memory was never the limiter. But same-repo **hard** batching at matched
  pool is the new best, ΔR@1 **+0.137** (R@1 0.515→0.652), with a tight Tier-2 CI
  ([+0.112,+0.164]; repo-cluster [+0.107,+0.173]; 31/32 held-out repos up) that closes
  R17a's open Tier-2-CI item.
- **Hardness confirmed multi-seed (R19):** the R18 gain is **not MPS noise**
  ([ablation-hardneg-confirm.md](ablation-hardneg-confirm.md)) — over seeds {0,1} the
  paired `repo-hard − random` ΔR@1 gap is +0.019 / +0.021 (mean **+0.020**, std ~0.001),
  above the ~±0.008 run-to-run noise; `repo-hard` averages **+0.142** (peak +0.146).
  Seed-0 reproduced R18's +0.1375 exactly, so the +0.146 first-run was a memory-leak
  artefact, not seed variance. Open: harder-negative **mining** (H2, code ready but
  deferred — its longer MPS cells exceeded the env's background-job limit), and a 3rd–5th
  seed when compute is steadier.
  And **Q6 follow-up:** a code-*embedding* base (not a code MLM — codebert
  collapsed; bge edged ahead). [ablation-code.md](ablation-code.md)

### Track E — Beyond MVP-1 (entry STARTED)
- **GraphRAG packer + SLM dry-run DONE (R12C):** relation-conditioned subgraph
  retrieval packs a context that a small SLM consumes — fixing PR in top-5 for
  18/20 sample issues; a 0.5B SLM grounds 2/3 (no benchmark claim yet).
  [slm-dryrun.md](slm-dryrun.md).
- **Next:** a trained relational SLM (QLoRA) with relation/policy heads (review,
  test selection, risk) + outcome learning. Still gated behind a solid retrieval
  win, which Track A now provides.

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
