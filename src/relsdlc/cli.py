"""``relsdlc`` command-line entry point.

Subcommands:

    relsdlc validate [PATH ...]   Validate records, edges, and cards (default: data/ schemas/).
    relsdlc bench [--task T]      Run the fixture retrieval benchmark and print metrics.
    relsdlc schemas              List the available schemas.

Exit code is non-zero when validation finds any error, so this doubles as a CI gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .schemas import SCHEMA_NAMES, schemas_dir
from .validate import Report, validate_path


def _default_targets() -> list[Path]:
    targets = [p for p in (Path("data"),) if p.exists()]
    return targets or [Path(".")]


def _cmd_validate(args: argparse.Namespace) -> int:
    targets = [Path(p) for p in args.paths] if args.paths else _default_targets()
    merged = Report()
    for target in targets:
        if not target.exists():
            print(f"ERROR: path not found: {target}", file=sys.stderr)
            return 2
        report = validate_path(target)
        merged.findings.extend(report.findings)
        merged.n_records += report.n_records
        merged.n_edges += report.n_edges
        merged.n_cards += report.n_cards
        merged.n_benchmarks += report.n_benchmarks

    if args.json:
        print(json.dumps({
            "ok": merged.ok,
            "counts": {"records": merged.n_records, "edges": merged.n_edges,
                       "cards": merged.n_cards, "benchmarks": merged.n_benchmarks},
            "errors": len(merged.errors),
            "warnings": len(merged.warnings),
            "findings": [f.__dict__ for f in merged.findings],
        }, indent=2))
    else:
        for finding in merged.findings:
            stream = sys.stderr if finding.severity == "error" else sys.stdout
            print(finding, file=stream)
        print(
            f"validated {merged.n_records} records, {merged.n_edges} edges, "
            f"{merged.n_cards} cards, {merged.n_benchmarks} benchmark queries: "
            f"{len(merged.errors)} error(s), {len(merged.warnings)} warning(s)"
        )
    return 0 if merged.ok else 1


def _cmd_bench(args: argparse.Namespace) -> int:
    # Imported lazily so `relsdlc validate` has no dependency on the bench path.
    from .bench import run_fixture_benchmark

    fixtures = Path(args.fixtures)
    if not (fixtures / "records.jsonl").exists():
        print(f"ERROR: no records.jsonl under {fixtures}", file=sys.stderr)
        return 2
    report = run_fixture_benchmark(fixtures, task=args.task)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for task_id, m in report["tasks"].items():
            recalls = " ".join(f"R@{k}={v:.3f}" for k, v in m["recall_at_k"].items())
            print(f"{task_id}: n={m['n_queries']} {recalls} "
                  f"MRR={m['mrr']:.3f} HardNegAcc={m['hard_negative_accuracy']:.3f}")
        if report["leakage"]:
            print(f"LEAKAGE: {len(report['leakage'])} violation(s): "
                  f"{', '.join(report['leakage'])}", file=sys.stderr)
    return 1 if report["leakage"] else 0


def _cmd_schemas(args: argparse.Namespace) -> int:
    print(f"schemas dir: {schemas_dir()}")
    for name in SCHEMA_NAMES:
        print(f"  - {name}.schema.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="relsdlc", description=__doc__)
    parser.add_argument("--version", action="version", version=f"relsdlc {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="validate records, edges, and cards")
    p_val.add_argument("paths", nargs="*", help="files or directories (default: data/)")
    p_val.add_argument("--json", action="store_true", help="emit JSON report")
    p_val.set_defaults(func=_cmd_validate)

    p_bench = sub.add_parser("bench", help="run the fixture retrieval benchmark")
    p_bench.add_argument("--fixtures", default="data/fixtures",
                         help="fixtures directory (default: data/fixtures)")
    p_bench.add_argument("--task", default=None, help="restrict to one task id")
    p_bench.add_argument("--json", action="store_true", help="emit JSON report")
    p_bench.set_defaults(func=_cmd_bench)

    p_sch = sub.add_parser("schemas", help="list available schemas")
    p_sch.set_defaults(func=_cmd_schemas)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
