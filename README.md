# Relational SDLC AI Lab

Public research lab for relational-geometric learning over software development
lifecycle records.

The project explores a simple thesis:

> Software engineering agents should learn and operate over verifiable
> relations among records, not only over semantically similar text chunks.

The first target is not a full autonomous coding agent. The first target is a
measurable relational retrieval layer for SDLC tasks:

- issue -> likely fixing PRs;
- diff -> affected tests;
- failing log -> likely files/functions;
- PR -> risk and missing-test candidates;
- task state -> tools or agents that should run next.

Contribution and source-record rules are in [CONTRIBUTING.md](CONTRIBUTING.md).
The public operating boundary is in [docs/operating-boundary.md](docs/operating-boundary.md).
The research lifecycle is in [docs/research-lifecycle.md](docs/research-lifecycle.md).

This public repository is intentionally limited to research notes, source
rules, dataset documentation, model designs, evaluation designs, and
reproducible experiment results. Local planning files, run state, and caches
should stay outside this repository.

## Initial North Stars

- **G1: Public research hygiene.** Public sources and reproducible records.
- **G2: Falsifiable retrieval gains.** Every claim needs a benchmark.
- **G3: Dataset provenance.** Every record has source, license, timestamp,
  and transformation lineage.
- **G4: Relation-aware models.** Embeddings and rerankers should learn explicit
  software lifecycle relations, not only text similarity.
- **G5: Agentic readiness.** Every workflow should eventually expose state,
  preconditions, actions, tools, and outcomes.

## First 90 Days

1. Build a 20-repo public dataset with issues, PRs, commits, diffs, tests, and
   relation edges.
2. Establish vanilla embedding baselines for issue/file/test retrieval.
3. Train a relation-aware embedding model with contrastive and relation-head
   objectives.
4. Evaluate Recall@K, MRR, hard-negative accuracy, and test-selection quality.
5. Publish dataset cards, model cards, experiment cards, and reproducibility
   notes.
6. Keep local lifecycle state outside public history.

## Quickstart

```bash
pip install -e ".[dev]"

relsdlc validate data     # schema + provenance + referential integrity + leakage gates
relsdlc bench             # Recall@K / MRR / hard-negative accuracy per task (synthetic fixture)
relsdlc ablation          # relation-supervised retrieval vs vanilla vs IDF (the headline result)
pytest -q                 # metrics, baseline, model, and validation regression tests

python data/fixtures/build_fixtures.py   # regenerate the synthetic fixture + example cards
python data/synth/build_synth.py         # regenerate the relation benchmark + ablation cards
```

### Headline result (synthetic, exploratory)

On a controlled benchmark where the `fixes` link is carried by rare tokens and the
surface is dominated by misleading common tokens, relation supervision decisively
beats vanilla text similarity (`relsdlc ablation`, 94 held-out queries):

| System | Recall@1 | MRR | Hard-neg accuracy |
|---|---|---|---|
| vanilla text cosine | 0.11 | 0.28 | 0.11 |
| unsupervised IDF | 0.38 | 0.50 | 0.38 |
| **relation-supervised** | **0.82** | **0.86** | **0.82** |

This demonstrates the *mechanism*, not a real-world result — see
[docs/ablation.md](docs/ablation.md).

### Real-data check (honest negative)

On the [P1 pilot](docs/ablation-real.md) — a frozen snapshot of 20 permissive
GitHub repos, real issue→fixing-PR (`python data/pilot/run_real_ablation.py`, 144
held-out queries) — the synthetic win **does not transfer**:

| System | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| vanilla text cosine | 0.35 | 0.72 | 0.52 |
| **unsupervised IDF** | **0.44** | **0.81** | **0.61** |
| relation-metric (diagonal) | 0.42 | 0.79 | 0.60 |

Real PRs restate the issues they fix, so surface similarity is already strong and
**IDF is the baseline to beat** — the diagonal relation model ties it, it does not
win.

We then made the testbed honest — **references removed** (no link to follow) and
**repos held out** (cross-repo generalization) — and added a two-tower projection
relation embedder. Result ([docs/ablation-crossrepo.md](docs/ablation-crossrepo.md)):
the tower **wins on cross-token synthetic (R@1 0.83) but loses to vanilla
cross-repo (0.24)** — bag-of-tokens projections can't transfer to unseen repos'
vocabulary; IDF (no training) stays the robust winner. The evidence-backed
conclusion: **cross-repo generalization needs pretrained semantic embeddings as
the feature substrate**.

### Embeddings settle the substrate ([docs/ablation-embed.md](docs/ablation-embed.md))

On the same de-referenced cross-repo split, a frozen small embedder (MiniLM-L6)
**wins outright**, and a bolt-on relation head can't improve on it at pilot scale:

| System | R@1 | R@5 | MRR |
|---|---|---|---|
| IDF (bag-of-tokens bar) | 0.46 | 0.83 | 0.62 |
| **embedder-cosine** | **0.59** | 0.92 | 0.73 |
| embedder + from-scratch head | 0.19 | 0.64 | 0.38 |
| embedder + identity-init operator | 0.59 | 0.93 | 0.74 |

So: **embeddings for the substrate** (settled); a frozen-embedder head adds nothing
(a from-scratch one actively overfits). The relational contribution therefore has
to live **inside the representation** (fine-tune with the relation loss) or **in
graph structure cosine can't see** (link prediction over the typed SDLC graph) —
the next experiments. The lab's deliverable is the frozen public benchmark + honest
baselines that make exactly this measurable.

The synthetic `datebox` fixture (CC0, original) lets the whole pipeline run from a
clean checkout with no network. It mirrors the timezone-bug worked example from
the position paper. Benchmark tasks and metrics are defined in
[docs/benchmark-definition.md](docs/benchmark-definition.md).

## Public Repo Contents

```text
schemas/    JSON schemas for records, edges, benchmark queries, and cards
src/        the relsdlc library + CLI (validation, metrics, baseline, benchmark)
data/       dataset cards, small public fixtures, and redistribution notes
docs/       architecture, roadmap, benchmark definition, and research notes
tests/      regression tests for schema, validation, metrics, and baseline
```

Local coordination state — planning files, operating playbooks, run logs,
caches, checkpoints — stays out of public history per
[docs/operating-boundary.md](docs/operating-boundary.md).

The substrate is the methodology made executable: every dataset record carries
provenance, every relation edge an extraction method and confidence, every claim
an experiment card, and every benchmark a frozen split with a leakage guard.
