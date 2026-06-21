"""Locate and load the JSON schemas under ``schemas/``.

Schemas live at the repository root (not inside the package) so they are easy to
read and cite from issues, cards, and docs. This module finds that directory by
walking up from the current working directory and from this file's location, then
caches the parsed schemas.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

SCHEMA_NAMES = (
    "provenance",
    "record",
    "edge",
    "benchmark-query",
    "source-card",
    "dataset-card",
    "experiment-card",
)


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    here = Path(__file__).resolve()
    # src/relsdlc/schemas.py -> repo root is three parents up.
    roots.append(here.parents[2])
    roots.append(Path.cwd())
    # Walk up from cwd in case the tool is invoked from a subdirectory.
    for parent in Path.cwd().resolve().parents:
        roots.append(parent)
    return roots


@functools.lru_cache(maxsize=1)
def schemas_dir() -> Path:
    """Return the directory containing ``*.schema.json``."""
    for root in _candidate_roots():
        candidate = root / "schemas"
        if (candidate / "record.schema.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate the schemas/ directory. Run relsdlc from within the "
        "relational-sdlc-ai-lab repository."
    )


@functools.lru_cache(maxsize=None)
def load_schema(name: str) -> dict:
    """Load one schema by short name (e.g. 'record', 'edge', 'source-card')."""
    path = schemas_dir() / f"{name}.schema.json"
    if not path.is_file():
        raise FileNotFoundError(f"Schema not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def all_schemas() -> dict[str, dict]:
    """Load every known schema, keyed by short name."""
    return {name: load_schema(name) for name in SCHEMA_NAMES}
