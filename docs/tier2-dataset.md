# Tier-2 dense dataset (Track D)

**Status: exploratory, real public data.** The 20-repo pilot (`data/pilot/`) and
the 55-repo "Tier-2 entry" (`data/scale/`, [`docs/scale-dataset.md`](scale-dataset.md))
established a robust finding: on real public issue→fixing-PR retrieval,
**unsupervised IDF weighting reliably beats plain cosine**, the diagonal
relation-metric only *ties* IDF (it does not win at bag-of-tokens scale), and a
LoRA fine-tune **does** beat the frozen embedder. This wave (R16B) grows the
dataset both **wider** (78 repos) and, deliberately, **denser per repo** — the
prior snapshots averaged ~10 fixes-queries/repo; this one averages **~35** by
ingesting more issue/PR pages per repo. The question: does the finding hold when
each repo contributes many more examples?

The frozen pilot and scale snapshots are **not touched** — other experiments
depend on them. The Tier-2 ids use a distinct `gh-t2:owner/repo:…` namespace and
are kept **id-disjoint** from both prior snapshots (a test pins this), so
`relsdlc validate data` — which validates the whole tree at once — stays clean.

## Counts

| | repos | records | edges (`fixes`) | benchmark queries | q / repo |
|---|---|---|---|---|---|
| pilot (`data/pilot/`) | 20 | 2,087 | 356 | 356 | ~18 |
| scale (`data/scale/`) | 55 | 3,672 | 562 | 562 | ~10 |
| **tier2 (`data/tier2/`)** | **78** | **16,998** (2,282 issue + 14,716 pull_request) | **2,744** | **2,744** | **~35** |

`records.jsonl` is ~17 MB. Because this snapshot is larger than the ~8 MB
single-file convention, the builder applies a defensive size cap and prunes
records down to those actually referenced by an edge or benchmark query. The
embeddings caches (`data/tier2/embeddings/*.npz`) are **gitignored** — regenerate
them with the finetune script below.

## Density, not just breadth

The headline design choice for this wave is **density**. Going from ~10 to ~35
fixes-queries per repo means the cross-repo evaluation sees a much richer per-repo
distribution of issue↔PR pairs, which is the regime that matters for representation
learning (LoRA): more positive pairs per repo to contrast against same-repo hard
negatives. 78 repos × ~35 queries gives **2,744** benchmark queries, ~4.9× the
scale entry's 562.

## Re-confirmed baseline (de-referenced, cross-repo)

Same protocol as the pilot/scale cross-repo ablation
([`docs/ablation-crossrepo.md`](ablation-crossrepo.md)): explicit cross-references
(`#N`, `gh-N`, URLs, SHAs) are **scrubbed** from the text so the model can't
string-match the issue number, and train repos are held **disjoint** from test
repos so a win must **generalize to unseen repositories**. `seed 0`, numpy-only
(no torch). 46 train repos / 32 test repos (1,573 train queries, 1,171 test
queries, vocab 5,641 at `min_df=8`).
[`data/tier2/run_tier2_ablation.py`](../data/tier2/run_tier2_ablation.py); raw
numbers in `data/tier2/tier2-results.json`.

| system | R@1 | R@5 | R@10 | MRR | HardNegAcc |
|---|---|---|---|---|---|
| vanilla-tf-cosine | 0.287 | 0.567 | 0.862 | 0.435 | 0.289 |
| **idf-cosine** | **0.389** | **0.713** | **0.913** | **0.542** | **0.392** |

**Why no diagonal relation-metric row here.** On both the pilot and the scale
entry the diagonal (reweight-shared-tokens-only) metric **ties** IDF — it does not
beat it — and that finding is already documented
([`docs/ablation-real.md`](ablation-real.md),
[`docs/scale-dataset.md`](scale-dataset.md)). At Tier-2's ~17k records the metric's
dense triplet stacks (triplets × vocab) cost more memory/time than the
already-settled result is worth, so the Tier-2 baseline runs **vanilla + IDF
only**. The metric remains available via `run_ablation(..., include_metric=True)`
for anyone who wants to re-confirm it on a smaller corpus.

## Re-confirmed LoRA fine-tune (de-referenced, cross-repo)

Same recipe as pilot/scale: LoRA (r=8, α=16) on the MiniLM-L6 attention q/k/v,
symmetric InfoNCE over **train-repo** fixes pairs only, 12 epochs; both the frozen
and tuned embedders are scored by raw cosine on the **held-out** test repos with
references scrubbed. [`data/tier2/finetune_tier2.py`](../data/tier2/finetune_tier2.py)
+ [`data/tier2/run_tier2_finetune.py`](../data/tier2/run_tier2_finetune.py); raw
numbers in `data/tier2/tier2-finetune-results.json`. 1,171 held-out test queries.

| system | R@1 | R@5 | R@10 | MRR | HardNegAcc |
|---|---|---|---|---|---|
| frozen MiniLM-L6 | 0.515 | 0.829 | 0.969 | 0.655 | 0.523 |
| **LoRA-tuned** | **0.629** | **0.921** | **0.985** | **0.757** | **0.635** |
| **Δ** | **+0.114** | +0.092 | +0.016 | **+0.101** | +0.112 |

**The LoRA win grows with density.** The cross-repo ΔR@1 is the largest in the
program so far — **+0.114**, up from **+0.080** at 55-repo scale (R13A) and **+0.07**
at the 20-repo pilot. Density (≈35 vs ≈10 fixes-queries/repo) gives the relation
loss more positive pairs per repo to contrast, and the held-out-repo gain widens
accordingly. MRR +0.101 and hard-negative accuracy +0.112 move together with R@1;
R@5/R@10 are already near-saturated so their headroom is smaller.

## Read

- **The robust finding holds at dense Tier-2.** Unsupervised IDF beats plain
  cosine on R@1 by **+0.102** (0.389 vs 0.287) on held-out repositories, and leads
  on R@5, R@10, MRR, and hard-negative accuracy too. The frequency signal
  generalizes — now across 78 repos with much denser per-repo coverage.
- **Density did not erase the gap.** Going from ~10 to ~35 queries/repo kept the
  IDF-over-vanilla margin intact (≈+0.10 R@1, consistent with pilot and scale),
  evidence the effect is structural to issue↔fix language, not a small-sample
  artifact.
- **The LoRA fine-tune win is the representation-learning story** — see the table
  above. A diagonal reweighting can't recover the issue↔fix relation beyond corpus
  IDF; learning the representation can.

## How to regenerate

The dataset is a **one-time live snapshot** (it touches the GitHub REST API and is
**not** reproducible — live data moves). CI never runs the builder; CI validates
the committed snapshot and re-runs the deterministic numpy ablation.

```bash
# 1. (re)build the snapshot — needs a token; live, polite, paced, rate-limit-aware
GITHUB_TOKEN=$(gh auth token) python data/tier2/build_tier2.py

# 2. validate the whole data tree (0 errors; warnings are fine)
PYTHONPATH=src python -m relsdlc.cli validate data

# 3. re-run the deterministic numpy baseline (writes cards + tier2-results.json)
PYTHONPATH=src python data/tier2/run_tier2_ablation.py

# 4. (torch) frozen + LoRA caches, then the LoRA-vs-frozen eval
python data/tier2/finetune_tier2.py        # needs the [embed] extra; writes gitignored caches
python data/tier2/run_tier2_finetune.py    # numpy eval -> tier2-finetune-results.json

# 5. the committed-snapshot tests
python -m pytest tests/test_tier2.py tests/test_tier2_lora.py -q
```

The builder is a polite guest: authenticated GitHub REST (~5000 req/hr), a
`User-Agent`, rate-limit-aware, paced; it fetches only **closed** issues + **closed**
PRs metadata (~5 pages, `per_page=100` — denser than scale's ~2 pages), truncates
bodies to 500 chars, mines `fixes` edges from closing keywords, builds same-repo
hard negatives by title/body token overlap, and freezes the split by repository.
Repos that error or are non-permissive are skipped; every record and edge carries
full provenance (real `sha256` content hash, never `TODO`).

## A JSONL robustness scar (fixed this wave)

The dense Tier-2 ingest surfaced a latent bug the smaller snapshots never hit: a PR
body contained a Unicode line separator (`U+2028`). The JSONL readers used
`str.splitlines()`, which treats `U+2028`/`U+2029`/`U+0085`/`\v`/`\f`/lone `\r` as
line boundaries — so a single record was split mid-string and `json.loads` raised
`Unterminated string`. Fixed by splitting on `"\n"` only across **all** JSONL
readers (library + every data script + every test helper), pinned by
[`tests/test_jsonl_robustness.py`](../tests/test_jsonl_robustness.py). The only
remaining `str.splitlines()` is a genuine non-JSONL description split.

## Artifacts

- `data/tier2/build_tier2.py` — live ingest (one-time snapshot).
- `data/tier2/{records,edges,source-cards}.jsonl`, `data/tier2/split.json`,
  `data/tier2/benchmark/issue_to_fixing_pr.jsonl` — the frozen dataset.
- `data/cards/examples/gh-tier2-v0.dataset-card.json` — dataset card.
- `data/tier2/run_tier2_ablation.py` — numpy-only baseline; writes
  `data/tier2/tier2-results.json` and the `gh-tier2-{vanilla,idf}-v0` experiment cards.
- `data/tier2/finetune_tier2.py` + `data/tier2/run_tier2_finetune.py` — torch LoRA
  fine-tune + numpy eval; writes `data/tier2/tier2-finetune-results.json` and the
  `gh-tier2-lora-v0` card.
- `tests/test_tier2.py`, `tests/test_tier2_lora.py` — committed-snapshot tests
  (validates clean, larger than scale, id-disjoint, IDF R@1 ≥ vanilla, LoRA > frozen).
