# Tier-2-entry scale dataset (Track D)

**Status: exploratory, real public data.** The 20-repo pilot (`data/pilot/`)
established a robust finding: on real public issue→fixing-PR retrieval,
**unsupervised IDF weighting reliably beats plain cosine**, while the
diagonal relation-metric ties IDF (it does not win at bag-of-tokens scale). The
obvious next question is whether that result holds with *more* data. This wave
(R12A) grows the dataset to a **~55-repo "Tier-2 entry"** under `data/scale/`
and re-confirms the bag-of-tokens baselines (vanilla / IDF / diagonal) with numpy.

The frozen pilot is **not touched** — other experiments depend on it. The scale
ids stay in the `gh:owner/repo:…` namespace and are kept **id-disjoint** from the
pilot snapshot (any colliding id is dropped at build time), so `relsdlc validate
data` — which validates the whole tree at once — stays clean.

## Counts

| | repos | records | edges (`fixes`) | benchmark queries |
|---|---|---|---|---|
| pilot (`data/pilot/`) | 20 | 2,087 | 356 | 356 |
| **scale (`data/scale/`)** | **55** | **3,672** (479 issue + 3,193 pull_request) | **562** | **562** |

`records.jsonl` is ~3.6 MB (well under the ~8 MB committed-snapshot ceiling).
The temporal-by-issue-created split (frozen, seed 0, earliest 60% of fixes =
train) yields **293 train / 186 test** issues.

## Repositories (55, permissive)

55 permissive, test-heavy, active repos — the pilot's 20 plus ~35 more from the
wider Python ecosystem and a few permissive JS/Go/Rust projects. License mix
(via GitHub's `/license` SPDX classification): **MIT 25, BSD-3-Clause 15,
Apache-2.0 4, "unknown" 11.** The 11 "unknown" are well-known permissive projects
whose `LICENSE` GitHub does not auto-classify to a single SPDX id (e.g. numpy's
BSD-variant, pyca/cryptography's Apache-OR-BSD dual license, matplotlib's PSF-style
license, mypy, sympy, trio, packaging, certifi, networkx, hypothesis, Pillow).
Only **metadata** is redistributed here (titles + bodies truncated to 500 chars),
never source, so the redistribution status is `metadata_only` for every source.

<details>
<summary>Full repo list with license</summary>

| repo | license |
|---|---|
| Delgan/loguru | MIT |
| HypothesisWorks/hypothesis | unknown |
| Textualize/rich | MIT |
| Textualize/textual | MIT |
| agronholm/anyio | MIT |
| aio-libs/aiohttp | Apache-2.0 |
| arrow-py/arrow | Apache-2.0 |
| astral-sh/ruff | MIT |
| certifi/python-certifi | unknown |
| dask/dask | BSD-3-Clause |
| encode/httpcore | BSD-3-Clause |
| encode/httpx | BSD-3-Clause |
| encode/starlette | BSD-3-Clause |
| encode/uvicorn | BSD-3-Clause |
| ewels/rich-click | MIT |
| fastapi/fastapi | MIT |
| fastapi/typer | MIT |
| jd/tenacity | Apache-2.0 |
| lxml/lxml | BSD-3-Clause |
| matplotlib/matplotlib | unknown |
| more-itertools/more-itertools | MIT |
| mwaskom/seaborn | BSD-3-Clause |
| networkx/networkx | unknown |
| numpy/numpy | unknown |
| pallets/click | BSD-3-Clause |
| pallets/flask | BSD-3-Clause |
| pallets/jinja | BSD-3-Clause |
| pallets/werkzeug | BSD-3-Clause |
| pandas-dev/pandas | BSD-3-Clause |
| psf/black | MIT |
| psf/requests | Apache-2.0 |
| pyca/cryptography | unknown |
| pydantic/pydantic | MIT |
| pydantic/pydantic-core | MIT |
| pypa/packaging | unknown |
| pypa/pip | MIT |
| pypa/setuptools | MIT |
| pypa/virtualenv | MIT |
| pypa/wheel | MIT |
| pytest-dev/pytest | MIT |
| python-attrs/attrs | MIT |
| python-pillow/Pillow | unknown |
| python-poetry/poetry | MIT |
| python-trio/trio | unknown |
| python/mypy | unknown |
| samuelcolvin/dirty-equals | MIT |
| scikit-learn/scikit-learn | BSD-3-Clause |
| scrapy/scrapy | BSD-3-Clause |
| sdispater/pendulum | MIT |
| sqlalchemy/sqlalchemy | MIT |
| sympy/sympy | unknown |
| theskumar/python-dotenv | BSD-3-Clause |
| tox-dev/tox | MIT |
| urllib3/urllib3 | MIT |
| yaml/pyyaml | MIT |

</details>

## Re-confirmed baseline (de-referenced, cross-repo)

Same protocol as the pilot's cross-repo ablation
([`docs/ablation-crossrepo.md`](ablation-crossrepo.md)): explicit cross-references
(`#N`, `gh-N`, URLs, SHAs) are scrubbed from the text so the model can't
string-match the issue number, and train repos are held disjoint from test repos
so a win must **generalize to unseen repositories**. `min_df=3`, seed 0,
numpy-only (no torch). 20 train repos / 14 test repos (337 train queries, 225
test queries, vocab 4,883).
[`data/scale/run_scale_ablation.py`](../data/scale/run_scale_ablation.py); raw
numbers in `data/scale/scale-results.json`.

| system | R@1 | R@5 | R@10 | MRR | HardNegAcc |
|---|---|---|---|---|---|
| vanilla-tf-cosine | 0.333 | 0.676 | 0.889 | 0.505 | 0.338 |
| **idf-cosine** | **0.444** | **0.778** | 0.911 | **0.599** | **0.449** |
| relation-metric | 0.422 | 0.729 | **0.920** | 0.572 | 0.427 |

## Read

- **The robust finding holds at scale.** Unsupervised IDF beats plain cosine on
  R@1 by **+0.111** (0.444 vs 0.333) on held-out repositories — a larger margin
  than typical at pilot scale — and IDF leads on R@5, MRR, and hard-negative
  accuracy too. The frequency signal generalizes across more, more-diverse repos.
- **The diagonal relation-metric still ties IDF, it does not beat it.** As
  documented for the pilot ([`docs/ablation-real.md`](ablation-real.md)), a
  diagonal (reweight-shared-tokens-only) metric cannot recover the issue↔fix
  relation beyond what corpus IDF already captures. Recovering that relation
  needs representation learning (LoRA fine-tuning) — see below.
- **This wave re-confirms the bag-of-tokens baselines and grows the data.**
  Embeddings / LoRA at scale are a **torch follow-up** (the relevant tracks are
  [`docs/ablation-finetune.md`](ablation-finetune.md),
  [`docs/ablation-scale.md`](ablation-scale.md), and
  [`docs/ablation-gnn.md`](ablation-gnn.md), which on the pilot showed the LoRA
  win is robust across 5 held-out-repo splits and stacks with the graph). Re-
  running those at this Tier-2 scale is the next step on Track D.

## How to regenerate

The dataset is a **one-time live snapshot** (it touches the GitHub REST API and
is **not** reproducible — live data moves). CI never runs the builder; CI
validates the committed snapshot and re-runs the deterministic ablation.

```bash
# 1. (re)build the snapshot — needs a token; live, polite, paced, rate-limit-aware
GITHUB_TOKEN=$(gh auth token) python data/scale/build_scale.py

# 2. validate the whole data tree (0 errors; warnings are fine)
PYTHONPATH=src python -m relsdlc.cli validate data

# 3. re-run the deterministic numpy ablation (writes cards + scale-results.json)
PYTHONPATH=src python data/scale/run_scale_ablation.py

# 4. the committed-snapshot tests
python -m pytest tests/test_scale.py -q
```

The builder is a polite guest: authenticated GitHub REST (~5000 req/hr), a
`User-Agent`, rate-limit-aware, paced; it fetches only **closed** issues +
**closed** PRs metadata (~2 pages, `per_page=100`), truncates bodies to 500
chars, mines `fixes` edges from closing keywords, builds same-repo hard
negatives by title/body token overlap, and freezes a temporal split. Repos that
error (404 / moved / empty) are skipped; the snapshot is capped to stay under the
~8 MB `records.jsonl` budget. Every record and edge carries full provenance
(real `sha256` content hash, never `TODO`).

## Artifacts

- `data/scale/build_scale.py` — live ingest (one-time snapshot).
- `data/scale/{records,edges,source-cards}.jsonl`, `data/scale/split.json`,
  `data/scale/benchmark/issue_to_fixing_pr.jsonl` — the frozen dataset.
- `data/cards/examples/gh-scale2-v0.dataset-card.json` — dataset card.
- `data/scale/run_scale_ablation.py` — numpy-only re-confirmation; writes
  `data/scale/scale-results.json` and `data/cards/examples/gh-scale2-*.experiment-card.json`.
- `tests/test_scale.py` — validates clean, substantially larger than the pilot,
  IDF R@1 ≥ vanilla R@1.
