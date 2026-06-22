# GraphRAG subgraph packer + small-SLM dry-run (MVP-2, Track E)

This is the **MVP-2 entry** from the [research roadmap](research-roadmap.md)
(Track E — "Relational SLM ... GraphRAG subgraph packer"). It is the first half
of the relational-SLM layer the [architecture](architecture.md) sketches:

```text
task -> relation-conditioned retrieval -> subgraph -> prompt/context -> response
```

There are two pieces, with very different epistemic status:

| Piece | File | Status |
|---|---|---|
| **GraphRAG subgraph packer** | [`src/relsdlc/subgraph.py`](../src/relsdlc/subgraph.py) | **Solid, CI-tested, deterministic** (numpy-only) |
| **Small-SLM triage dry-run** | [`data/pilot/slm_demo.py`](../data/pilot/slm_demo.py) | **Best-effort demo** — no benchmark, no quantitative claim |

## What "GraphRAG packing" is

Flat RAG retrieves the top-k *text chunks* most similar to a query and pastes
them into a prompt. **GraphRAG** instead retrieves a *subgraph*: it starts from a
similarity hit, then walks typed relations to pull in structurally-connected
nodes the text retriever cannot see, and serialises that subgraph as the model's
context. Here the relations are the SDLC graph's `fixes` (PR→issue) and
`modifies` (PR→file/test) edges.

## Relation-conditioned retrieval

Given an **issue** node, [`subgraph.py`](../src/relsdlc/subgraph.py) does three
deterministic steps:

1. **Retrieve.** Rank candidate PRs by cosine of their pretrained MiniLM node
   embeddings (the Track-A representation) against the issue's embedding; keep
   the top `k`. Ties break by node id, so the order is reproducible.
2. **Expand along relations.** For each retrieved PR, follow its `modifies` edges
   to the concrete **source files and tests** it changed. This is the structural
   context flat text RAG cannot represent — file/test nodes have *no text
   embedding at all*; they only exist in the graph. The issue's own `fixes`
   neighbours are pulled in too when the graph records them.
3. **Pack.** Serialise the issue + the related PRs (each with its changed files
   and tests, truncated to fixed character/count budgets) into one structured,
   length-bounded string ready to drop into an SLM prompt.

The packer is **pure numpy** — no torch, no network, no transformers — and a pure
function of its inputs. The same subgraph always packs byte-identically. It is
covered by [`tests/test_subgraph.py`](../tests/test_subgraph.py), which runs in
CI on the committed pilot data + embeddings (and skips if they are absent).

### Retrieval quality (committed pilot snapshot)

On the de-referenced `issue_to_fixing_pr` benchmark, restricting each query to its
candidate pool, the **true fixing PR appears in the top-5** for the large
majority of sample issues (≈0.9 on the first 40 queries). This is the same
embedding signal the Track-A LoRA ablation measured; the packer's contribution is
turning that ranking *plus the graph structure* into a model-ready context, not a
new retrieval model. The retrieval half is the load-bearing, evaluated deliverable.

## The SLM dry-run

[`slm_demo.py`](../data/pilot/slm_demo.py) feeds the packed GraphRAG context to a
**small instruct model** — `Qwen/Qwen2.5-0.5B-Instruct`, the smallest viable
instruct model (~0.5 B params, ~1 GB) — on CPU with greedy decoding and short
`max_new_tokens`, and asks for a short structured triage: likely fix, which files
to look at, which tests to run.

**This is a DRY-RUN demonstration, not a benchmark.** No quantitative claim is
made about the generation. It shows the *end-to-end shape* — relation-packed
context in, structured triage out — and nothing more. It is deliberately kept out
of CI (it needs torch + transformers + a one-time download); it is run once
locally to produce the committed samples under
[`data/pilot/slm-outputs/`](../data/pilot/slm-outputs/). If the model download or
CPU generation is infeasible in a given environment, the packer still runs and the
contexts are written with a "generation deferred" note — the retrieval half ships
regardless.

### Dry-run examples (committed samples)

For the trailing-newline bug (`issue:3577`), the packer retrieves the fixing PR
top-1 and expands it to `rich/text.py` + `tests/test_ansi.py`; the SLM's triage
correctly names both as the files to look at and the test to run:

> **Likely fix** — `Text.from_ansi()` removes trailing newline characters …
> **Files to look at** — `rich/text.py`, `tests/test_ansi.py`
> **Suggested tests** — `tests/test_ansi.py`

For the cell-width bug (`issue:3958`), the triage centres on `rich/cells.py` +
`tests/test_cells.py`, both drawn from the retrieved subgraph. For a third issue
(`issue:3947`) the 0.5 B model **invented** plausible-but-wrong file names
(`text.pyx`, `test_rich.py`) — an honest failure mode for a model this small, and
exactly why this is a dry-run with no quantitative claim. The full inputs and
generations are in [`data/pilot/slm-outputs/`](../data/pilot/slm-outputs/).

## What a *trained* relational SLM would add next

This dry-run uses an off-the-shelf instruct model with **no relational training**
— it only sees the packed context as plain text. The MVP-2/3 follow-ups are:

- **Relation/policy heads (QLoRA).** Fine-tune a small SLM on the SDLC graph with
  supervision for review, test-selection, and risk, so the model *learns* the
  relational structure rather than reading it as prose.
- **Grounded decoding / constrained file selection.** Force the "files to look
  at" / "suggested tests" outputs to be drawn from the retrieved subgraph's nodes,
  eliminating the hallucinated-path failure mode seen above.
- **Learned subgraph selection.** Replace the fixed cosine-top-k + 1-hop expansion
  with a learned retriever (the GNN / link-prediction track), and measure triage
  quality against a held-out ground truth — at which point a quantitative claim
  becomes appropriate.

Per the roadmap's sequencing, these come **only after** the retrieval and
evaluation gates are boring; this wave delivers the packer and a labelled,
honest dry-run, not a trained model.
