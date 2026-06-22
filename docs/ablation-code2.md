# Q6 follow-up: does a code-*embedding* base beat the general substrate?

**Status: exploratory, real public data.** This closes the open follow-up from
[ablation-code.md](ablation-code.md), which found the axis that matters is
*embedding-tuned vs not*, not *code vs general*: a code MLM (CodeBERT) **collapses**
for retrieval (R@1 0.14), while a strong embedding-tuned general model (bge-small)
edges past MiniLM. The unresolved question was a model that is BOTH code-aware AND
embedding-tuned. This tests it, on the same de-referenced, frozen, cross-repo split
(10 train repos / 8 held-out test repos, 174 queries, `issue_to_fixing_pr`).

## Which model loaded (and which failed)

Candidates were tried with standard `AutoModel`/`AutoTokenizer` under
**transformers 5.12.1 / torch 2.12.1**, preferring NO remote code:

| Model | code-aware | embed-tuned | loaded under transformers 5.x? |
|---|---|---|---|
| `microsoft/unixcoder-base` | yes | no (MLM/code base) | **yes** — standard `AutoModel`, no remote code |
| `flax-sentence-embeddings/st-codesearch-distilroberta-base` | yes | **yes** (contrastive on CodeSearchNet) | **yes** — standard `AutoModel`, no remote code |
| `jinaai/jina-embeddings-v2-base-code` | yes | yes | **NO** — remote-code blocker (below) |
| `Salesforce/codet5p-110m-embedding` | yes | yes | NO — remote-code mismatch (carried over from Q6) |

**Compatibility blocker (jina code embeddings):** its `trust_remote_code`
`modeling_bert.py` does `from transformers.pytorch_utils import
find_pruneable_heads_and_indices`, a symbol **removed in transformers 5.x**, so the
import fails before any forward pass:

```
ImportError: cannot import name 'find_pruneable_heads_and_indices'
             from 'transformers.pytorch_utils'
```

This is the same *class* of failure that sank codet5p-110m-embedding in Q6: a
remote-code embedding model pinned to a pre-5.x transformers internal. The two
models that load are exactly the two that use the *stock* architectures
(`unixcoder` = RoBERTa, `st-codesearch` = DistilRoBERTa) — no remote code, no
vendored `pytorch_utils` symbols.

**Sanity check (before embedding the pilot):** both loaded models separate related
from unrelated text — `cos(code, paraphrase)` ≫ `cos(code, unrelated)`
(unixcoder 0.46 vs 0.07; st-codesearch 0.44 vs 0.06), so neither suffers CodeBERT's
near-degenerate geometry.

## Result (frozen, 8 held-out repos, `issue_to_fixing_pr`, 174 queries)

| Base | code? | embed-tuned? | pooling | R@1 | R@5 | MRR |
|---|---|---|---|---|---|---|
| MiniLM-L6 (substrate) | no | yes | mean | **0.592** | 0.920 | 0.728 |
| bge-small-en-v1.5 (stronger general) | no | yes | CLS | **0.598** | **0.948** | **0.743** |
| **st-codesearch-distilroberta** (code + embed-tuned) | yes | yes | mean | 0.546 | 0.925 | 0.708 |
| unixcoder-base (code, not embed-tuned) | yes | no | mean | 0.454 | 0.845 | 0.626 |
| codebert-base (code MLM, from Q6) | yes | no | mean | 0.144 | 0.448 | 0.306 |

Cards: [`data/cards/examples/gh-code2-*.experiment-card.json`](../data/cards/examples/).
Results: [`data/pilot/code2-results.json`](../data/pilot/code2-results.json).

## Finding

A code-*embedding* base produces a **competitive but not winning** frozen embedding:

- **Embedding-tuning rescues a code base from collapse.** st-codesearch (code +
  contrastive) lands at R@1 0.546 / R@5 0.925 — far above the raw code bases
  (unixcoder 0.454, CodeBERT 0.144). Its R@5 (0.925) essentially **ties** the MiniLM
  substrate (0.920). This is the strongest confirmation yet of the Q6 axis: it is the
  *embedding objective*, not the code pretraining, that makes a usable retrieval vector.

- **But code-awareness does not beat a good general embedder here.** Even the
  code+embed-tuned model trails MiniLM (R@1 0.592) and bge-small (0.598) on R@1/MRR.
  unixcoder (code, no embed-tuning) sits between CodeBERT and the substrate — better
  than a raw MLM, still well short of a frozen general embedder.

- **The ordering is monotone in embedding-tuning, not in code-awareness:**
  `CodeBERT (0.14) < unixcoder (0.45) < st-codesearch (0.55) < MiniLM (0.59) ≈ bge (0.60)`.
  Adding code knowledge moves a model UP only when an embedding objective is also
  present, and even then it does not overtake the general substrate at this scale.

## Honest verdict

**No — a code-embedding base does not (yet) beat the general substrate on this
frozen pilot.** The best code-aware embedder (st-codesearch) is *competitive*
(R@5 ties; R@1 −0.046 vs MiniLM, −0.052 vs bge) but does not win. The Q6 thesis
holds tighter than before: embedding-tuning is the load-bearing axis; code
pretraining is, at best, neutral-to-helpful once embedding-tuned, and harmful when
not. Why a *general* model still edges out a *code* one is plausibly the issue text:
the queries are natural-language bug reports, where a general-English embedder has
the advantage and the code-specific weights buy little on the issue side.

## Implications & follow-ups

- **bge-small remains the cheap drop-in upgrade to the substrate** (still the top
  frozen base). Swapping in a code-embedding base is not justified by these numbers.
- The larger lever is still **LoRA fine-tuning** (ablation-embed.md, finding §3):
  on frozen vectors a bolt-on relation head has no headroom. The natural next
  experiment is **LoRA on bge-small** (or, to actually exploit code structure,
  LoRA on st-codesearch with the contrastive/relation loss applied to the
  *code/diff* side specifically, where code-awareness should pay off — the issue
  side is natural language).
- **Remote-code embedding models (jina-code, codet5p) need an older-transformers
  env.** Recommended follow-up: pin `transformers<5` (e.g. 4.4x) in a separate,
  isolated env to evaluate `jinaai/jina-embeddings-v2-base-code` and
  `Salesforce/codet5p-110m-embedding`, then commit their caches the same way — so
  the numpy eval stays transformers-version-agnostic on the committed npz.

## Reproduce

```
# Embed step needs the [embed] extra (transformers + torch); cache is committed.
python data/pilot/embed_code2.py            # writes embeddings/code2-*.npz
python data/pilot/run_code2_ablation.py     # numpy-only eval on the committed caches
```
