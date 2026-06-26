#!/usr/bin/env python3
"""R26 — sampled label-precision audit of the VCS-harvested relations (the last
[PENDING] empirical item).

The benchmark calls its links "verifiable." This audit quantifies that claim, honestly
separating two notions of precision:

  - CONSTRUCTION precision (programmatic, deterministic, over ALL edges): did the mining
    rule actually fire correctly?
      * `fixes` (PR->issue): the source PR's raw text contains a GitHub closing keyword
        (close/fix/resolve [+s/d]) referencing the target issue number.
      * `modifies` (PR->test): the target is a real test path (`_is_test_path`) and the
        source is a PR.
    Construction precision is an UPPER BOUND on semantic precision.

  - SEMANTIC relatedness PROXY (for `fixes`): token-Jaccard between the issue title and
    the PR title. NOT ground truth — a weak signal — reported only as a distribution, with
    a rendered 20-edge sample for manual rating.

Reads the raw (UN-scrubbed) records so the closing keywords are still present. numpy-free,
deterministic. Run: PYTHONPATH=src python data/pilot/run_label_audit.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from relsdlc.ingest import _is_test_path  # noqa: E402

OUT = HERE / "label-audit-results.json"
SAMPLE_N = 20
# close/closes/closed, fix/fixes/fixed, resolve/resolves/resolved  +  [:|space] [#|GH-] <num>
_CLOSE = re.compile(r"(?i)\b(?:clos(?:e|es|ed)|fix(?:es|ed)?|resolv(?:e|es|ed))\b[\s:]+(?:#|gh-)?(\d+)")
_TOKEN = re.compile(r"[a-z0-9]+")


def _jl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]


def _num(node_id, rec):
    if rec and isinstance(rec.get("content"), dict) and rec["content"].get("number") is not None:
        return str(rec["content"]["number"])
    return node_id.split(":")[-1]


def _title(rec):
    c = rec.get("content") if rec else None
    return str(c.get("title", "")) if isinstance(c, dict) else ""


def _text(rec):
    c = rec.get("content") if rec else None
    if isinstance(c, dict):
        return f"{c.get('title', '')}\n{c.get('body', '')}"
    return str(c or "")


def _toks(s):
    return set(_TOKEN.findall(s.lower()))


def _jaccard(a, b):
    ta, tb = _toks(a), _toks(b)
    return len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0


def main():
    recs = {r["id"]: r for r in _jl(HERE / "records.jsonl")}
    fixes = [e for e in _jl(HERE / "edges.jsonl") if e.get("relation") == "fixes"]
    mods = [e for e in _jl(HERE / "graph" / "modifies_edges.jsonl") if e.get("relation") == "modifies"]

    # ---- fixes: construction precision (closing keyword references target issue) ----
    fx_ok, fx_checkable, jacc, sample = 0, 0, [], []
    for e in fixes:
        src, tgt = recs.get(e["source"]), recs.get(e["target"])
        if not src:
            continue
        fx_checkable += 1
        want = _num(e["target"], tgt)
        nums = set(_CLOSE.findall(_text(src)))
        hit = want in nums
        fx_ok += hit
        j = _jaccard(_title(tgt), _title(src)) if tgt else 0.0
        jacc.append(j)
        if len(sample) < SAMPLE_N:
            sample.append({
                "pr": e["source"], "issue": e["target"],
                "closing_keyword_found": bool(hit),
                "issue_title": _title(tgt)[:90], "pr_title": _title(src)[:90],
                "title_jaccard": round(j, 3)})

    # ---- modifies: construction precision = the edge correctly asserts "PR changed this
    # file" (source is a PR, target is a file/test node). Whether the file is a TEST is a
    # graph-COMPOSITION statistic, NOT a precision number — report it separately.
    md_ok = sum(1 for e in mods
                if ":pr:" in e["source"] and re.search(r":(file|test):", e["target"]))
    md_test_frac = sum(1 for e in mods if _is_test_path(e["target"].split(":")[-1]))

    jacc_sorted = sorted(jacc)
    med = jacc_sorted[len(jacc_sorted) // 2] if jacc_sorted else 0.0
    res = {
        "fixes": {
            "n_edges": len(fixes), "n_checkable": fx_checkable,
            "construction_precision": round(fx_ok / fx_checkable, 4) if fx_checkable else None,
            "definition": "source PR text contains a closing keyword referencing the target issue number",
            "method_distribution": dict(Counter(e.get("provenance", {}).get("method") for e in fixes)),
            "semantic_proxy_title_jaccard": {
                "median": round(med, 4),
                "frac_overlap_gt_0": round(sum(1 for j in jacc if j > 0) / len(jacc), 4) if jacc else None,
                "note": "weak proxy, NOT ground truth; manual rating needed for true semantic precision"},
        },
        "modifies": {
            "n_edges": len(mods),
            "construction_precision": round(md_ok / len(mods), 4) if mods else None,
            "definition": "edge correctly asserts 'PR changed this file' (source is a PR, target a file/test node); ~1.0 by git-history construction",
            "test_file_fraction": round(md_test_frac / len(mods), 4) if mods else None,
            "test_file_fraction_note": "share of modify-edges whose target is a TEST path — a graph-composition statistic, NOT a precision number"},
        "sample_for_manual_rating": sample,
        "honest_note": ("Construction precision is an UPPER BOUND on semantic precision. "
                        "`fixes` semantic precision (did the PR actually fix the issue) and "
                        "`modifies` semantic precision (is the test actually exercised by the diff) "
                        "require human raters; the sample above is provided for that."),
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    f, m = res["fixes"], res["modifies"]
    print("R26 — label-precision audit")
    print(f"  fixes:    construction precision {f['construction_precision']:.1%} "
          f"({fx_ok}/{fx_checkable} edges have a closing keyword referencing the target issue)")
    print(f"            method dist: {f['method_distribution']}")
    print(f"            semantic proxy (issue/PR title Jaccard): median {f['semantic_proxy_title_jaccard']['median']:.3f}, "
          f"{f['semantic_proxy_title_jaccard']['frac_overlap_gt_0']:.0%} have >0 overlap")
    print(f"  modifies: construction precision {m['construction_precision']:.1%} "
          f"({md_ok}/{len(mods)} edges correctly assert PR->file)")
    print(f"            test-file fraction {m['test_file_fraction']:.1%} (composition stat, not precision)")
    print(f"  wrote {OUT} (incl. {len(sample)}-edge sample for manual rating)")


if __name__ == "__main__":
    main()
