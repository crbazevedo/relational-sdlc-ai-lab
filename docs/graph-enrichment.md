# Pilot Graph Enrichment (`modifies` edges + file/test nodes)

The first pilot snapshot ([`data/pilot/`](../data/pilot/)) carried `issue`,
`pull_request`, and `commit` records plus `fixes` edges mined from PR closing
keywords. It had **no file-level structure** — no `file`/`test` nodes and no
`modifies` edges — so the GNN / link-prediction track and the diff→affected-test
task had nothing to learn over.

This enrichment adds that structure. For each **distinct fixing-PR** in
[`data/pilot/edges.jsonl`](../data/pilot/edges.jsonl) whose PR record exists in
[`data/pilot/records.jsonl`](../data/pilot/records.jsonl), it fetches the PR's
changed files (`GET /repos/{owner}/{repo}/pulls/{number}/files`) and maps them
into `file` / `test` records and `modifies` edges, using the same
provenance-bearing helpers as the R4 ingest tooling
([`src/relsdlc/ingest.py`](../src/relsdlc/ingest.py): `map_changed_file`,
`extract_modifies_edges`).

## What was fetched (committed snapshot)

| Quantity | Count |
|---|---|
| Fixing-PRs queried | 348 |
| PRs failed / skipped (404 / 410) | 0 |
| File records | 497 |
| Test records | 239 |
| File + test records (total) | 736 |
| `modifies` edges | 1356 |
| Repos covered | 18 |

The output lives under [`data/pilot/graph/`](../data/pilot/graph/), file-disjoint
from the original pilot records/edges:

- `file_records.jsonl` — the new `file` / `test` records (deduped, sorted by id).
- `modifies_edges.jsonl` — the `modifies` edges (sorted by source/relation/target).

Each `modifies` edge's `source` is an existing PR record id (so it joins the
graph the original pilot already committed) and its `target` is a `file`/`test`
record written here. The 239 `test` records make the diff→affected-test task
feasible: a PR that touches both source files and test files gives a directly
observed (PR → test) supervision signal.

## Provenance

Every file record and `modifies` edge carries full provenance: `source_url`
(the public GitHub blob / PR-files URL), `retrieved_at`, `license` (reused from
the matching [source card](../data/pilot/source-cards.jsonl); `unknown` when the
upstream license could not be resolved), a real `sha256:` `content_hash` (never
`TODO`), `transform`, `method` (`git_history` — observed directly from the diff),
and `observed: true`. `modifies` edges carry `confidence: 1.0` (observed from the
diff itself). Redistribution stays **`metadata_only`** — only short file paths
are redistributed, kept verbatim (no truncation needed); no file contents.

### Temporal consistency

File records get an early `valid_from` (`2024-01-01T00:00:00Z`) so they predate
every edge that touches them. Each `modifies` edge's `valid_from` is the **later**
of its source PR's `valid_from` and the file's `valid_from`, so the edge never
precedes either endpoint — no `temporal.inconsistent` warning fires for the
enriched graph.

## How to regenerate (one-time live snapshot)

This is a **live, one-time snapshot** — it touches the network and is not
reproducible (live data moves). CI never fetches; it validates the committed
output. To regenerate:

```bash
GITHUB_TOKEN=$(gh auth token) PYTHONPATH=src python3 data/pilot/build_graph.py
```

The script is pure standard library (`urllib` / `json`) and reads the token from
the `GITHUB_TOKEN` environment variable — it never hardcodes or prints it.

### Politeness and rate limits

The fetch is polite by construction (it reuses the R4 ingest request helpers):

- Sets a descriptive `User-Agent` and pins `X-GitHub-Api-Version: 2022-11-28`.
- Honours rate-limit headers: when `X-RateLimit-Remaining` hits `0` it sleeps
  until `X-RateLimit-Reset`.
- Paginates modestly (`per_page=100`, capped at 3 pages per PR — file paths are
  short and PRs rarely touch more) and pauses ~0.35 s between page requests.

Authenticated GitHub REST allows ~5000 requests/hour; the ~348 PR-file calls in
this snapshot stay well within that budget. A few PR file-fetches can fail
(404 / 410 for deleted PRs); the script skips them gracefully and reports the
skipped count (0 in the committed snapshot).

## Validation

The committed graph is checked the same way as the rest of the dataset:

```bash
PYTHONPATH=src python3 -m relsdlc.cli validate data    # 0 errors (warnings ok)
python3 -m pytest -q tests/test_graph.py               # hermetic, offline
```

[`tests/test_graph.py`](../tests/test_graph.py) is hermetic (offline): it asserts
the committed graph validates clean (0 errors), every `modifies` edge source
resolves to a PR record and every target resolves to a file record, and at least
one `test` record is present. It skips if the snapshot is absent.
