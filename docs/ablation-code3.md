# Q6 finish: a TRUE code-EMBEDDING base — under a pinned `transformers<5` env

**Status: exploratory, real public data.** This finishes Q6. The earlier rounds
([ablation-code.md](ablation-code.md), [ablation-code2.md](ablation-code2.md))
found the axis that matters is *embedding-tuned vs not*, not *code vs general*, but
left one stone unturned: the genuine code-*embedding* models
(`jinaai/jina-embeddings-v2-base-code`, `Salesforce/codet5p-110m-embedding`) would
not load — they **fail under transformers 5.x**, and the pilot/scale machine ships
transformers 5.12. This round pins `transformers<5` in an isolated venv, loads one
of those models, embeds the same de-referenced, frozen, cross-repo split (10 train
repos / 8 held-out test repos, 174 queries, `issue_to_fixing_pr`), and compares it
to the frozen MiniLM (0.592) and bge (0.598) baselines.

## The compatibility blocker, and the pinned-env fix

R13B / ablation-code2.md recorded the failure precisely: the jina-code remote-code
`modeling_bert.py` does

```
from transformers.pytorch_utils import find_pruneable_heads_and_indices
```

a symbol **removed in transformers 5.x**, so the import dies before any forward
pass (`ImportError: cannot import name 'find_pruneable_heads_and_indices'`).
`Salesforce/codet5p-110m-embedding` fails the same *class* of way (remote code
pinned to a pre-5.x internal). These are exactly the models we wanted, and exactly
the ones a 5.x environment cannot run.

The fix is environment, not code: pin `transformers<5` so the symbol is restored,
while reusing the system torch 2.x.

### Recipe (reproduce)

```bash
# Pinned venv — reuse system torch 2.x, pin transformers<5 in the venv.
uv venv .venv-r15b --python /opt/homebrew/opt/python@3.14/bin/python3.14 --system-site-packages
uv pip install --python .venv-r15b 'transformers>=4.40,<5' einops sentencepiece

# Verify: torch 2.x (system) + transformers<5 (venv) + the symbol that was missing.
.venv-r15b/bin/python -c "import torch, transformers; print(torch.__version__, transformers.__version__); \
  from transformers.pytorch_utils import find_pruneable_heads_and_indices; print('symbol present')"

# Embed (pinned env). Caches a committed npz; downstream eval is numpy-only.
.venv-r15b/bin/python data/pilot/embed_code3_pinned.py

# Compare (system python, numpy only — version-agnostic on the committed npz).
python data/pilot/run_code3_ablation.py
```

**Pinned versions used here:** `--system-site-packages` worked — the venv's
`transformers<5` pin shadows the system 5.12, while torch is inherited:

| component | version | source |
|---|---|---|
| torch | 2.11.0 | system site-packages (inherited) |
| transformers | **4.57.6** | venv pin (`>=4.40,<5`) |
| einops | 0.8.2 | venv (jina-code requires it) |
| sentencepiece | 0.2.1 | venv |
| Python | 3.14.3 | `python@3.14` |

`--system-site-packages` did **not** leak transformers 5 — the venv install of
`transformers<5` takes precedence, and `find_pruneable_heads_and_indices` resolves.
So the isolated non-system fallback (let the venv pull its own torch) was not
needed. `embed_code3_pinned.py` refuses to run (SystemExit 3) if it ever detects
transformers 5.x, so the pin can't silently regress.

## Which model loaded

The embed script tries the genuine code-embedding models in order and uses the
first that loads AND passes a sanity probe (cos(related) clearly above
cos(unrelated)). The first candidate succeeded:

| Model | code-aware | embed-tuned | loaded under transformers<5? |
|---|---|---|---|
| **`jinaai/jina-embeddings-v2-base-code`** | yes | yes | **YES** — `trust_remote_code`, mean-pooled, needs einops |
| `Salesforce/codet5p-110m-embedding` | yes | yes | not reached (jina-code chosen first) |
| `nomic-ai/CodeRankEmbed` | yes | yes | not reached |

The blocker from ablation-code2.md is **resolved by the pin**: jina-code's remote
code imports cleanly under transformers 4.57.6.

**Sanity check (before embedding the pilot):** jina-code separates related from
unrelated text cleanly — `cos(code, paraphrase) = 0.739` vs
`cos(code, unrelated) = 0.017`. No trace of the CodeBERT near-degenerate geometry
(where unrelated text sat at cosine ≈ 0.97); this is a real retrieval embedder.

## Result (frozen, 8 held-out repos, `issue_to_fixing_pr`, 174 queries)

| Base | code? | embed-tuned? | pooling | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|---|
| MiniLM-L6 (substrate) | no | yes | mean | **0.592** | 0.920 | 0.989 | 0.728 |
| bge-small-en-v1.5 (stronger general) | no | yes | CLS | **0.598** | 0.948 | 1.000 | **0.743** |
| **jina-embeddings-v2-base-code** (code + embed-tuned) | yes | yes | mean | 0.580 | **0.960** | 0.994 | 0.742 |

Cards: [`data/cards/examples/gh-code3-*.experiment-card.json`](../data/cards/examples/).
Results: [`data/pilot/code3-results.json`](../data/pilot/code3-results.json).

## Finding

A genuine code-embedding base is **competitive — it wins R@5, ties MRR, narrowly
loses R@1 — but does not clearly beat the general substrate:**

- **It is the best R@5 of any base tested across all three Q6 rounds (0.960)** —
  above bge (0.948) and MiniLM (0.920). When a slightly larger candidate set is
  acceptable, the code-embedding base recovers the right PR most reliably.
- **Its MRR (0.742) effectively ties bge (0.743)** and beats MiniLM (0.728) —
  the ranking quality is on par with the strongest general embedder.
- **But on strict R@1 it trails (0.580)** — just under MiniLM (0.592) and bge
  (0.598). For top-1 exactness, the general embedders still edge it out.

Set against the full Q6 ordering, code-embedding finally lands *at the top tier*
rather than below it:

```
CodeBERT (0.14) < unixcoder (0.45) < st-codesearch (0.55) < jina-code (0.58) ≈ MiniLM (0.59) ≈ bge (0.60)   [R@1]
```

and on R@5 it actually **leads**: `jina-code 0.960 > bge 0.948 > st-codesearch 0.925 ≈ MiniLM 0.920`.
The earlier code+embed-tuned model (st-codesearch, loadable under 5.x) was
*competitive but trailing*; the genuine pinned code-embedding model is
*competitive and, on R@5/MRR, at-or-above the substrate*.

## Honest verdict

**Not a clean win, but the closest yet — and a win on two of three metrics.** A
true code-embedding base (`jina-embeddings-v2-base-code`), once you pay the
pinned-`transformers<5` cost to load it, **beats the general substrate on R@5
(0.960 vs 0.948/0.920) and ties it on MRR (0.742 vs 0.743/0.728), while losing R@1
by a hair (0.580 vs 0.592/0.598).** So the strict answer to Q6 — "does a true
code-embedding base beat the general substrate?" — is **a qualified yes on
recall@5 and MRR, no on recall@1.** The Q6 thesis holds: embedding-tuning is the
load-bearing axis (the code-embedding model sits in the top tier with the
embedding-tuned general models, far above the raw code MLMs), and code-awareness
adds a real edge on R@5 but does not dominate. The likely reason R@1 still favors
a general embedder is the query side: the queries are natural-language bug reports,
where general-English weights help on top-1 exactness even as code-awareness helps
gather the right candidate into the top-5.

## Implications & follow-ups

- **For top-5 retrieval, the code-embedding base is now the best frozen option.**
  For strict top-1, bge-small remains the cheap drop-in (no pinned env needed).
- The remaining lever is still **LoRA fine-tuning** (ablation-embed.md): a frozen
  base, code or general, has limited headroom. The natural next experiment is
  **LoRA on jina-code with the contrastive/relation loss applied to the code/diff
  side** — where the code-awareness that already won R@5 should pay off most.
- **The pinned-env recipe is now a reusable artifact.** Any future code-embedding
  model that needs a pre-5.x `pytorch_utils` surface can be evaluated the same way:
  pin `transformers<5`, embed, commit the npz, eval with numpy. The committed cache
  keeps the downstream eval transformers-version-agnostic, so CI never needs the
  pinned env.

## Reproduce

```bash
# Embed step needs the pinned transformers<5 env (see Recipe above); cache committed.
.venv-r15b/bin/python data/pilot/embed_code3_pinned.py     # writes embeddings/code3-*.npz
python data/pilot/run_code3_ablation.py                    # numpy-only eval on committed caches
```
