"""Regression: JSONL readers split on '\\n' only, not str.splitlines().

A record body can legitimately contain Unicode line separators (U+2028, U+2029,
U+0085 NEL, vertical tab, form feed). json.dumps(ensure_ascii=False) writes some of
these literally, so a reader that uses str.splitlines() would split a record
mid-string and raise JSONDecodeError. The dense Tier-2 ingest (R16B) hit exactly
this (a PR body with U+2028). These tests pin the '\\n'-only behavior.
"""

from __future__ import annotations

import json

from relsdlc.bench import load_jsonl
from relsdlc.validate import _iter_objects

# Characters str.splitlines() treats as line boundaries but JSONL must not.
TRICKY = "  \x85\x0b\x0c"


def _write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


def test_bench_load_jsonl_handles_unicode_line_separators(tmp_path):
    p = tmp_path / "recs.jsonl"
    _write(p, [{"id": "a", "body": f"line1{TRICKY}line2"}, {"id": "b", "body": "ok"}])
    rows = load_jsonl(p)
    assert len(rows) == 2
    assert rows[0]["id"] == "a" and TRICKY in rows[0]["body"]


def test_validate_iter_objects_handles_unicode_line_separators(tmp_path):
    p = tmp_path / "recs.jsonl"
    _write(p, [{"id": "x", "type": "issue", "content": {"body": f"a{TRICKY}b"}},
               {"id": "y", "type": "issue"}])
    objs = list(_iter_objects(p))
    assert len(objs) == 2  # not split into 3+ by the U+2028 in the body
