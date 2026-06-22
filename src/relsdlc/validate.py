"""Validation engine for records, edges, and cards.

Three layers of checks:

1. **Schema** — each object validates against its ``schemas/*.schema.json``.
2. **Provenance** — every record/edge carries a real (non-TODO) content hash;
   every edge records its extraction ``method``.
3. **Dataset integrity** — referential integrity (edge endpoints resolve to
   records), unique record ids, and temporal-consistency / leakage checks.

The engine returns structured findings so the CLI can exit non-zero on any
error while still printing every problem it found.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from .schemas import SCHEMA_NAMES, load_schema

_CARD_SCHEMA_BY_TYPE = {
    "source": "source-card",
    "dataset": "dataset-card",
    "experiment": "experiment-card",
}


@dataclass(frozen=True)
class Finding:
    severity: str  # "error" | "warning"
    code: str
    message: str
    location: str = ""

    def __str__(self) -> str:
        where = f" [{self.location}]" if self.location else ""
        return f"{self.severity.upper()}: {self.code}: {self.message}{where}"


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    n_records: int = 0
    n_edges: int = 0
    n_cards: int = 0
    n_benchmarks: int = 0

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


# --- schema registry ---------------------------------------------------------


def _registry() -> Registry:
    resources = []
    for name in SCHEMA_NAMES:
        schema = load_schema(name)
        resources.append((schema["$id"], Resource.from_contents(schema)))
    return Registry().with_resources(resources)


def _validator(schema_name: str) -> Draft202012Validator:
    return Draft202012Validator(load_schema(schema_name), registry=_registry())


def _schema_findings(obj: dict, schema_name: str, location: str) -> list[Finding]:
    findings: list[Finding] = []
    for err in sorted(_validator(schema_name).iter_errors(obj), key=str):
        path = "/".join(str(p) for p in err.absolute_path)
        loc = f"{location}:{path}" if path else location
        findings.append(Finding("error", "schema", err.message, loc))
    return findings


# --- helpers -----------------------------------------------------------------


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def classify(obj: dict) -> str:
    """Classify a loaded object as 'card', 'edge', 'benchmark', or 'record'."""
    if "card_type" in obj:
        return "card"
    if "relation" in obj:
        return "edge"
    if "query_record" in obj:
        return "benchmark"
    return "record"


# --- single-object validation ------------------------------------------------


def validate_record(obj: dict, location: str = "") -> list[Finding]:
    findings = _schema_findings(obj, "record", location)
    prov = obj.get("provenance", {})
    if prov.get("content_hash") == "TODO":
        findings.append(
            Finding("error", "provenance.todo",
                    "record content_hash is 'TODO'; committed records need a real sha256 hash",
                    location)
        )
    if prov.get("retrieved_at") and _parse_iso(prov["retrieved_at"]) is None:
        findings.append(
            Finding("error", "provenance.timestamp",
                    f"unparseable retrieved_at: {prov['retrieved_at']!r}", location)
        )
    return findings


def validate_edge(obj: dict, location: str = "") -> list[Finding]:
    findings = _schema_findings(obj, "edge", location)
    prov = obj.get("provenance", {})
    if not prov.get("method"):
        findings.append(
            Finding("error", "provenance.method",
                    "edge provenance.method is required (how was this relation extracted?)",
                    location)
        )
    if prov.get("content_hash") == "TODO":
        findings.append(
            Finding("error", "provenance.todo",
                    "edge content_hash is 'TODO'; committed edges need a real sha256 hash",
                    location)
        )
    if prov.get("observed") is False and isinstance(obj.get("confidence"), (int, float)):
        if obj["confidence"] > 0.9:
            findings.append(
                Finding("warning", "confidence.inferred",
                        f"inferred edge (observed=false) has high confidence {obj['confidence']}",
                        location)
            )
    return findings


def validate_benchmark(obj: dict, location: str = "") -> list[Finding]:
    findings = _schema_findings(obj, "benchmark-query", location)
    candidates = set(obj.get("candidates", []))
    for rel in obj.get("relevant", []):
        if rel not in candidates:
            findings.append(
                Finding("error", "benchmark.relevant",
                        f"relevant id {rel!r} is not in the candidate pool", location)
            )
    for neg in obj.get("hard_negatives", []):
        if neg not in candidates:
            findings.append(
                Finding("warning", "benchmark.hardneg",
                        f"hard_negative id {neg!r} is not in the candidate pool", location)
            )
    return findings


def validate_card(obj: dict, location: str = "") -> list[Finding]:
    card_type = obj.get("card_type")
    schema_name = _CARD_SCHEMA_BY_TYPE.get(card_type)
    if schema_name is None:
        return [Finding("error", "card.type",
                        f"unknown card_type {card_type!r}; expected one of "
                        f"{sorted(_CARD_SCHEMA_BY_TYPE)}", location)]
    return _schema_findings(obj, schema_name, location)


# --- dataset-level validation ------------------------------------------------


def validate_dataset(records: list[dict], edges: list[dict],
                     record_locs: list[str] | None = None,
                     edge_locs: list[str] | None = None,
                     benchmarks: list[dict] | None = None,
                     benchmark_locs: list[str] | None = None) -> list[Finding]:
    """Cross-object checks: unique ids, referential integrity, temporal consistency."""
    findings: list[Finding] = []
    record_locs = record_locs or [""] * len(records)
    edge_locs = edge_locs or [""] * len(edges)
    benchmarks = benchmarks or []
    benchmark_locs = benchmark_locs or [""] * len(benchmarks)

    seen: dict[str, str] = {}
    valid_from: dict[str, datetime] = {}
    for rec, loc in zip(records, record_locs):
        rid = rec.get("id")
        if rid is None:
            continue
        if rid in seen:
            findings.append(
                Finding("error", "id.duplicate",
                        f"duplicate record id {rid!r} (first at {seen[rid]})", loc)
            )
        seen[rid] = loc
        if rec.get("valid_from"):
            dt = _parse_iso(rec["valid_from"])
            if dt is not None:
                valid_from[rid] = dt

    for edge, loc in zip(edges, edge_locs):
        for endpoint in ("source", "target"):
            ref = edge.get(endpoint)
            if ref is not None and ref not in seen:
                findings.append(
                    Finding("error", "ref.dangling",
                            f"edge {endpoint} {ref!r} does not resolve to a record", loc)
                )
        edge_from = _parse_iso(edge.get("valid_from", "")) if edge.get("valid_from") else None
        if edge_from is not None:
            for endpoint in ("source", "target"):
                ref = edge.get(endpoint)
                ep_from = valid_from.get(ref)
                if ep_from is not None and edge_from < ep_from:
                    findings.append(
                        Finding("warning", "temporal.inconsistent",
                                f"edge valid_from precedes {endpoint} {ref!r} valid_from "
                                "(possible temporal leakage)", loc)
                    )

    for bench, loc in zip(benchmarks, benchmark_locs):
        refs = [bench.get("query_record")] + list(bench.get("candidates", []))
        for ref in refs:
            if ref is not None and ref not in seen:
                findings.append(
                    Finding("error", "ref.dangling",
                            f"benchmark query references unknown record {ref!r}", loc)
                )
    return findings


# --- file / directory driver -------------------------------------------------


def _iter_objects(path: Path):
    """Yield (obj, location) from a .json (object or list) or .jsonl file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        # Split on "\n" only — NOT str.splitlines(), which also breaks on Unicode
        # line separators ( / /\x85) that json.dumps(ensure_ascii=False)
        # writes literally inside a body, splitting a record mid-string.
        for lineno, line in enumerate(text.split("\n"), start=1):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line), f"{path.name}:{lineno}"
    else:
        data = json.loads(text)
        if isinstance(data, list):
            for i, obj in enumerate(data):
                yield obj, f"{path.name}[{i}]"
        else:
            yield data, path.name


# Auxiliary JSON that are not schema artifacts (splits, run results, etc.).
_SKIP_NAMES = {"split.json", "ablation-results.json"}


def _discover(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    files: list[Path] = []
    for pattern in ("*.json", "*.jsonl"):
        files.extend(sorted(target.rglob(pattern)))
    # Templates carry intentional TODO placeholders; never validate them.
    return [
        f for f in files
        if "templates" not in f.parts
        and f.name not in _SKIP_NAMES
        and not f.name.endswith("-results.json")
    ]


def validate_path(target: Path) -> Report:
    """Validate every record/edge/card found under ``target`` (file or directory)."""
    report = Report()
    records: list[dict] = []
    edges: list[dict] = []
    benchmarks: list[dict] = []
    record_locs: list[str] = []
    edge_locs: list[str] = []
    benchmark_locs: list[str] = []

    for path in _discover(target):
        # Skip schema files themselves.
        if path.suffix == ".json" and path.name.endswith(".schema.json"):
            continue
        try:
            objects = list(_iter_objects(path))
        except json.JSONDecodeError as exc:
            report.add(Finding("error", "json", f"invalid JSON: {exc}", path.name))
            continue
        for obj, loc in objects:
            if not isinstance(obj, dict):
                report.add(Finding("error", "shape", "expected a JSON object", loc))
                continue
            kind = classify(obj)
            if kind == "card":
                report.n_cards += 1
                for f in validate_card(obj, loc):
                    report.add(f)
            elif kind == "edge":
                report.n_edges += 1
                edges.append(obj)
                edge_locs.append(loc)
                for f in validate_edge(obj, loc):
                    report.add(f)
            elif kind == "benchmark":
                report.n_benchmarks += 1
                benchmarks.append(obj)
                benchmark_locs.append(loc)
                for f in validate_benchmark(obj, loc):
                    report.add(f)
            else:
                report.n_records += 1
                records.append(obj)
                record_locs.append(loc)
                for f in validate_record(obj, loc):
                    report.add(f)

    for f in validate_dataset(records, edges, record_locs, edge_locs,
                              benchmarks, benchmark_locs):
        report.add(f)
    return report
