"""R26 label-precision audit: pins the committed result so the "verifiable
relations" claim cannot silently degrade.

Pure JSON read; skips if absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RES = Path(__file__).resolve().parents[1] / "data" / "pilot" / "label-audit-results.json"

pytestmark = pytest.mark.skipif(not RES.exists(), reason="R26 label audit not present")


def _d():
    return json.loads(RES.read_text(encoding="utf-8"))


def test_fixes_construction_precision_high():
    f = _d()["fixes"]
    # closing-keyword construction precision is near-perfect
    assert f["construction_precision"] > 0.95
    assert f["n_checkable"] > 100


def test_modifies_construction_precision_perfect():
    m = _d()["modifies"]
    # git-history construction: every edge correctly asserts a PR->file change
    assert m["construction_precision"] == 1.0
    # the test-file fraction is reported as composition, NOT precision
    assert 0.0 < m["test_file_fraction"] < 1.0


def test_sample_is_present_for_manual_rating():
    assert len(_d()["sample_for_manual_rating"]) >= 10
