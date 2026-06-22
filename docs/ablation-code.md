# Q6: does the embedding base matter? (it's "embedding-tuned", not "code")

**Status: exploratory, real public data.** Q6 from the roadmap: would a code-aware
base beat the general MiniLM substrate? Tested on the same de-referenced cross-repo
split, frozen (no fine-tune), so it isolates the base.

## Result (frozen, 8 held-out repos, issue_to_fixing_pr)

| Base | pooling | R@1 | R@5 | MRR |
|---|---|---|---|---|
| MiniLM-L6 (substrate) | mean | 0.592 | 0.920 | 0.728 |
| **bge-small-en-v1.5** (stronger general embedder) | CLS | **0.598** | **0.948** | **0.743** |
| codebert-base (code-pretrained, MLM) | mean | 0.144 | 0.448 | 0.306 |

Cards: [`data/cards/examples/gh-code-*.experiment-card.json`](../data/cards/examples/).

## Finding

The axis that matters is **embedding-tuned vs not**, not **code vs general**:

- A strong *embedding-tuned* general model (bge-small) edges past MiniLM
  (+0.015 MRR, +0.028 R@5) — base quality gives a small lift for free.
- A *code-pretrained but not embedding-tuned* model (CodeBERT, a masked-LM)
  **collapses** to R@1 0.14 — its mean-pooled vectors barely separate (unrelated
  texts sit at cosine ≈ 0.97). Code knowledge in the weights does not make a good
  retrieval embedding.

## Implication

The real Q6 follow-up is a **code-*embedding* model** (a model embedding-tuned on
code, e.g. a CodeT5+-embedding / code-contrastive model) — not a raw code MLM. A
compatible one wasn't loadable in this environment (remote-code/version mismatch),
so it is queued. Meanwhile, bge-small is a cheap drop-in upgrade to the substrate,
and the LoRA fine-tune (which reshapes whatever base it starts from) remains the
larger lever. Combine: LoRA on bge-small at Track-D scale is the natural next step.
