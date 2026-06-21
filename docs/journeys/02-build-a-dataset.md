# Journey 02 — Build a dataset

This journey takes you from a research idea to a small, validated dataset with
full provenance. The discipline mirrors the
[research lifecycle](../research-lifecycle.md): intake first, then curate
records and relation edges, then prove the data is sound with
`relsdlc validate data`.

Everything here is **metadata + small public fixtures only**. Raw large data and
local run state stay out of the repository per the
[operating boundary](../operating-boundary.md).

## 1. Intake — record the research frame before collecting anything

Before writing a single record, write down the frame. This is the
[Research Intake](../research-lifecycle.md#1-research-intake) step:

- **hypothesis** — the falsifiable claim, e.g. *"relation-trained embeddings
  retrieve the fixing PR for an issue better than vanilla text embeddings."*
- **target task** — one of the four benchmark tasks (see
  [benchmark definition](../benchmark-definition.md)), e.g. `issue_to_fixing_pr`.
- **allowed / excluded source classes** — e.g. permissively licensed public
  repos with issues linked to PRs and tests present; exclude restricted or
  unlicensed sources.
- **baseline** — the dependency-light `baseline-hashing-tfidf` floor.
- **target metrics** — Recall@K, MRR, hard-negative accuracy.
- **expected public outputs** — source card, dataset card, experiment card.
- **known risks** — sampling bias, label noise, leakage.

Keep this intake note wherever your planning lives; the *public* outputs of it
are the cards you write below.

## 2. Write a source card

For each source you ingest, author a **source card** so the data is traceable.
Start from the template at
[`data/cards/templates/source-card.template.json`](../../data/cards/templates/source-card.template.json):

```json
{
  "card_type": "source",
  "id": "src:<owner>-<repo>",
  "name": "<repo display name>",
  "source_url": "https://github.com/<owner>/<repo>",
  "retrieved_at": "TODO-ISO-8601",
  "license": "<SPDX id, e.g. MIT, Apache-2.0>",
  "terms_note": "<API terms / rate-limit / redistribution constraints>",
  "record_types": ["issue", "pull_request", "commit", "diff", "file", "test", "ci_log"],
  "transform": "<reproducible ingest command>",
  "content_hash": "TODO",
  "redistribution": "metadata_only",
  "notes": "<why this repo qualifies>"
}
```

Fill in `retrieved_at` (ISO-8601), the SPDX `license`, the exact reproducible
`transform` command, and a real `content_hash` (`sha256:<hex>`). `TODO` for the
hash is permitted **only in templates** — never in a committed record.

The committed example
[`data/cards/examples/datebox.source-card.json`](../../data/cards/examples/datebox.source-card.json)
shows a filled-in source card for the synthetic fixture.

## 3. Add records with provenance

A record is a node in the relational graph. Its shape is fixed by
[`schemas/record.schema.json`](../../schemas/record.schema.json) — required
fields are `id`, `type`, and `provenance`. The `type` is a closed enum:
`issue`, `pull_request`, `commit`, `diff`, `file`, `symbol`, `test`, `ci_log`,
`tool`, `agent_run`.

Records live as JSON Lines (one object per line). For example, an issue record
from the fixture:

```json
{"id": "issue:482", "type": "issue", "content": {"title": "Date filter returns incorrect results when timezone is UTC-3", "body": "The date range filter normalizes timezone incorrectly for UTC-3; filtered results are off by one day."}, "provenance": {"source_url": "synthetic://datebox", "retrieved_at": "2024-02-01T00:00:00Z", "license": "CC0-1.0", "content_hash": "sha256:a8c6fa52c7790888ebc9635dfc7b6c4592e4e77db3a9784dfea6dd230d899c91", "method": "synthetic", "observed": true, "transform": "python data/fixtures/build_fixtures.py"}, "valid_from": "2024-01-05T00:00:00Z"}
```

Every record needs **provenance**
([`schemas/provenance.schema.json`](../../schemas/provenance.schema.json)) with
the required `source_url`, `retrieved_at`, `license`, and `content_hash`. Add
`valid_from` (the commit/timestamp at which the record becomes valid) so the
temporal-leakage guard can do its job later.

> For a real dataset you would not hand-write these — you would author a
> reproducible ingest script (the `transform` command) that emits the JSONL.
> The synthetic fixture is built by
> [`data/fixtures/build_fixtures.py`](../../data/fixtures/build_fixtures.py);
> use it as a reference for emitting records deterministically.

## 4. Add relation edges with provenance

Edges are the typed relations between records — the part that makes this a
*relational* dataset. Their shape is fixed by
[`schemas/edge.schema.json`](../../schemas/edge.schema.json) — required fields
are `source`, `relation`, `target`, `confidence`, and `provenance`. The
`relation` is a closed enum that includes `fixes`, `modifies`, `covers`,
`caused_by`, `fails_on`, and more.

The `fixes` edge that ties the worked example together:

```json
{"source": "pr:512", "relation": "fixes", "target": "issue:482", "confidence": 1.0, "provenance": {"source_url": "synthetic://datebox", "retrieved_at": "2024-02-01T00:00:00Z", "license": "CC0-1.0", "content_hash": "sha256:27f7b9c0bd8089f6299e468b6badefd9712cdd7584f10d640168611c6e521f85", "method": "synthetic", "observed": true}, "valid_from": "2024-01-08T00:00:00Z"}
```

Guidance for edges:

- **`confidence`** is a weight in `[0, 1]`. Directly observed edges (e.g. a
  PR-closes-issue link from `git_history`) deserve high confidence; automatically
  mined edges should reflect their noise with lower confidence.
- **`method`** records the extraction channel (`static_analysis`,
  `test_coverage`, `git_history`, `review`, `ci_log`, `human_label`,
  `agent_trace`, `synthetic`).
- Both `source` and `target` must resolve to real record ids — referential
  integrity is enforced by `validate`.

## 5. Write a dataset card

Roll the records and edges up into a **dataset card** from
[`data/cards/templates/dataset-card.template.json`](../../data/cards/templates/dataset-card.template.json).
Record the counts, the source ids, and — critically — the **split policy**:

```json
"split_policy": {
  "frozen": true,
  "method": "temporal-by-commit-date",
  "seed": 0,
  "boundary": "<cutoff date or commit>"
}
```

For real data the default split method is **temporal-by-commit-date** (train on
the past, evaluate on the future) — random splits leak. See the filled example
[`data/cards/examples/datebox-fixture-v0.dataset-card.json`](../../data/cards/examples/datebox-fixture-v0.dataset-card.json).

## 6. Validate (the boring gate)

```bash
relsdlc validate data
```

On the current fixture you should see:

```text
validated 15 records, 10 edges, 3 cards, 3 benchmark queries: 0 error(s), 0 warning(s)
```

`validate` checks schema validity, provenance completeness, referential
integrity, and the temporal-leakage guard, and **exits non-zero on any error**,
so it doubles as a CI gate. If you want machine-readable output for tooling:

```bash
relsdlc validate data --json
```

To confirm which schemas are in play:

```bash
relsdlc schemas
```

Expected output:

```text
schemas dir: <repo>/schemas
  - provenance.schema.json
  - record.schema.json
  - edge.schema.json
  - benchmark-query.schema.json
  - source-card.schema.json
  - dataset-card.schema.json
  - experiment-card.schema.json
```

## 7. Before you publish

Run the [pre-publish checklist](../operating-boundary.md#before-publishing): a
source record for every data point, a frozen benchmark or an explicit
`exploratory` label, no restricted source material, no local planning or run
state, no credentials or local paths, and every claim tied to a metric,
experiment card, or public source.

Once the data validates, you are ready to benchmark it — continue to
[Journey 03](03-run-the-benchmark.md).
