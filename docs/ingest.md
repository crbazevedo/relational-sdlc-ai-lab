# Public GitHub Ingest

This document describes the offline-runnable, live-gated ingest that maps public
GitHub data into the [record](../schemas/record.schema.json) /
[edge](../schemas/edge.schema.json) schema with full
[provenance](../schemas/provenance.schema.json).

The ingest tooling lives in [`src/relsdlc/ingest.py`](../src/relsdlc/ingest.py)
and is pure standard library (`urllib`, `json`, `hashlib`, `argparse`) — it adds
no dependencies. It is **offline by default**: the default code path reads a
RECORDED API snapshot and never opens a socket. Live fetch is opt-in and gated
behind an explicit `--live` flag plus a `GITHUB_TOKEN`.

## What it produces

| GitHub source | Record / edge | Method |
|---|---|---|
| issue JSON | `issue` record | provenance only |
| PR JSON | `pull_request` record | provenance only |
| commit JSON | `commit` record | provenance only |
| PR's changed file | `file` (or `test`) record | — |
| "Fixes/Closes #N" in PR body | `fixes` edge | `human_label` |
| PR's changed files | `modifies` edge | `git_history` |

Every record and edge carries full provenance: `source_url`, `retrieved_at`,
`license`, `content_hash` = `sha256:` of the canonical JSON of the content,
`transform`, `method`, and `observed`. Edges record a `confidence` in `[0, 1]`
(human-authored closing keywords get `0.95`; files observed directly from a diff
get `1.0`). A closing-keyword reference to an issue that is not in the snapshot is
**not** minted as an edge, so referential integrity holds.

Test files (paths under `tests/`, `*_test.*`, `*.test.*`, `*.spec.*`, etc.) are
typed as `test` records rather than `file`.

## Run it (offline — the default)

Transform the recorded snapshot under `tests/ingest_fixtures/` into schema records
and edges, and write a source card:

```bash
python -m relsdlc.ingest transform \
    --fixtures tests/ingest_fixtures \
    --out data/ingest_example \
    --source-card

relsdlc validate data        # the produced example passes every gate
```

The committed example under [`data/ingest_example/`](../data/ingest_example/) is
exactly this transform's output, so it is reproducible from the recorded
fixtures with no network. A regression test
([`tests/test_ingest.py`](../tests/test_ingest.py)) asserts the committed example
matches the transform.

## Run it live (opt-in)

Live fetch requires `--live`. Without it, `fetch` refuses and points you at the
offline `transform` path. Set `GITHUB_TOKEN` for the higher rate limit:

```bash
GITHUB_TOKEN=ghp_xxx python -m relsdlc.ingest fetch \
    --repo owner/name \
    --out tests/ingest_fixtures \
    --live

# then transform the recorded snapshot offline:
python -m relsdlc.ingest transform --fixtures tests/ingest_fixtures --out data/ingest_example
```

`fetch` records the raw API responses to a `snapshot.json`; `transform` then maps
that snapshot into schema records and edges. Keeping fetch and transform separate
means the network step is recorded once and the mapping is replayable offline.

### Politeness and rate limits

The live path is polite by construction:

- Sets a descriptive `User-Agent` header on every request.
- Sends `Accept: application/vnd.github+json` and pins
  `X-GitHub-Api-Version: 2022-11-28`.
- Honours rate-limit headers: when `X-RateLimit-Remaining` hits `0` it sleeps
  until `X-RateLimit-Reset`.
- Pauses briefly between page requests.

Authenticated GitHub REST is limited to roughly **5000 requests/hour**;
unauthenticated requests are limited to about **60/hour**, so a token is strongly
recommended for any non-trivial pull.

### Redistribution defaults

The default redistribution policy for ingested GitHub data is **`metadata_only`**.
Source cards record the upstream license (resolved from the repo's
`/license` endpoint when fetching live). Records carry provenance back to the
public GitHub URLs; full source text is not redistributed in this public repo.
Only redistribute snippets or full text when the upstream license clearly permits
it, and reflect that choice in the source card's `redistribution` field.

## The 20-repo selection criteria

The first-90-days goal is a 20-repo public dataset (README G-targets, item 1).
Candidate repos should satisfy:

1. **Permissive license** (MIT / Apache-2.0 / BSD) so metadata and, where the
   license allows, snippets can be redistributed.
2. **Active pull requests** — a healthy stream of merged PRs to mine `modifies`
   and `fixes` relations from.
3. **Issues linked to PRs** — closing keywords ("Fixes #N", "Closes #N") present
   in PR bodies so `fixes` edges can be extracted with `human_label` method.
4. **Tests present** — a real test suite so `test` records and (later) `covers`
   relations are derivable.
5. **Manageable size** — large enough to be interesting, small enough to ingest
   within polite rate limits and keep public fixtures small.
6. **Multiple languages** — a mix so retrieval gains are not language-specific.

A candidate shortlist spanning languages and ecosystems:

- Python: `pytest`, `fastapi`, `pydantic`, `requests`
- JS/TS: `vite`, `eslint`, `prettier`, `express`
- Rust: `ripgrep`
- Go: `cobra`, `zap`
- Observability / infra libraries: `prometheus` client libraries

Each selected repo gets a source card before any record derived from it enters a
dataset (the `source` card is a hard prerequisite per
[`docs/operating-boundary.md`](operating-boundary.md)).

## Hermetic example

The offline example in this repo ingests a small, hand-authored,
GitHub-REST-shaped snapshot (an example `octo-demo/widgets` repo) recorded under
[`tests/ingest_fixtures/`](../tests/ingest_fixtures/). The raw API JSON lives
under `tests/` (not `data/`) so `relsdlc validate data` never tries to validate
raw API responses — only the transformed, schema-conformant output under
`data/ingest_example/` is validated.
