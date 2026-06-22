"""Benchmark harness: run a retrieval task over a record set and score it.

A benchmark query is a JSON object:

    {
      "query_id": "...",
      "task": "issue_to_fixing_pr",
      "query_record": "<record id>",
      "candidates": ["<record id>", ...],
      "relevant": ["<record id>", ...],
      "hard_negatives": ["<record id>", ...],   # optional
      "as_of": "2024-01-10T00:00:00Z"            # optional; enables leakage check
    }

The query text is the text view of ``query_record``; candidate texts are the
text views of the candidate records. The default system is the vanilla baseline
embedder, so this harness measures the floor that relation-aware models beat.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path

from . import baseline, metrics

Ranker = Callable[[str, Sequence[str], Mapping[str, str]], list[str]]


def record_text(record: dict) -> str:
    """Flatten a record's content into a single retrievable text view."""
    parts: list[str] = [str(record.get("id", "")), str(record.get("type", ""))]
    content = record.get("content", {})
    if isinstance(content, dict):
        for key in ("title", "name", "path", "message", "body", "summary",
                    "code", "log", "text", "diff"):
            val = content.get(key)
            if isinstance(val, str):
                parts.append(val)
        # Include any remaining short string fields.
        for key, val in content.items():
            if key not in {"title", "name", "path", "message", "body", "summary",
                           "code", "log", "text", "diff"} and isinstance(val, str):
                parts.append(val)
    elif isinstance(content, str):
        parts.append(content)
    return "\n".join(p for p in parts if p)


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    # Split on "\n" only (JSONL delimiter); str.splitlines() also breaks on Unicode
    # line separators that can appear literally inside a record body.
    for line in path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def leakage_violations(query: dict, records_by_id: Mapping[str, dict]) -> list[str]:
    """Return ids of candidates/positives that are not yet valid at the query's as_of time."""
    as_of_raw = query.get("as_of")
    if not as_of_raw:
        return []
    as_of = _parse_iso(as_of_raw)
    if as_of is None:
        return []
    violations: list[str] = []
    seen: set[str] = set()
    for cid in list(query.get("candidates", [])) + list(query.get("relevant", [])):
        if cid in seen:
            continue
        seen.add(cid)
        rec = records_by_id.get(cid)
        if not rec or not rec.get("valid_from"):
            continue
        vf = _parse_iso(rec["valid_from"])
        if vf is not None and vf > as_of:
            violations.append(cid)
    return violations


def run_queries(queries: Sequence[dict], records_by_id: Mapping[str, dict],
                ranker: Ranker | None = None) -> tuple[list[metrics.RetrievalResult], list[str]]:
    ranker = ranker or baseline.rank_against
    texts = {rid: record_text(rec) for rid, rec in records_by_id.items()}
    results: list[metrics.RetrievalResult] = []
    leakage: list[str] = []
    for q in queries:
        qtext = texts.get(q["query_record"], "")
        ranked = ranker(qtext, list(q["candidates"]), texts)
        results.append(metrics.RetrievalResult.of(
            ranked, q.get("relevant", []), q.get("hard_negatives", [])))
        for bad in leakage_violations(q, records_by_id):
            leakage.append(f"{q.get('query_id', '?')}:{bad}")
    return results, leakage


def run_fixture_benchmark(fixtures_dir: Path, task: str | None = None,
                          ks: Sequence[int] = (1, 5, 10)) -> dict:
    """Load the fixture dataset + benchmark queries and score each task."""
    records = load_jsonl(fixtures_dir / "records.jsonl")
    records_by_id = {r["id"]: r for r in records}

    bench_dir = fixtures_dir / "benchmark"
    query_files = sorted(bench_dir.glob("*.jsonl"))
    report: dict = {"tasks": {}, "leakage": []}
    for qf in query_files:
        queries = load_jsonl(qf)
        if task:
            queries = [q for q in queries if q.get("task") == task]
        if not queries:
            continue
        results, leakage = run_queries(queries, records_by_id)
        task_id = queries[0].get("task", qf.stem)
        report["tasks"][task_id] = metrics.evaluate(results, ks)
        report["leakage"].extend(leakage)
    return report
