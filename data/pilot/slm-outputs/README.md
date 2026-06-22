# SLM dry-run outputs (MVP-2, Track E)

Committed samples of the **GraphRAG → small-SLM dry-run** (`data/pilot/slm_demo.py`). The deterministic subgraph packer (`src/relsdlc/subgraph.py`) is the CI-tested deliverable; this directory holds best-effort generation samples only.

**Status.** Generations were produced once locally with `Qwen/Qwen2.5-0.5B-Instruct` on CPU (greedy decoding). They are committed samples of a DRY-RUN demo — there is no benchmark or quantitative claim attached to them.

Each `gh_*.md` file shows, for one sample issue: the relation-conditioned packed context (issue + top-k related PRs with their changed files/tests) and the SLM's structured triage (or a deferral note).

See [`docs/slm-dryrun.md`](../../../docs/slm-dryrun.md) for the write-up.
