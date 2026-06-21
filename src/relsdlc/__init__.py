"""Relational SDLC AI Lab core library.

Public, dependency-light building blocks for relational-geometric learning over
software-development-lifecycle records:

- ``relsdlc.schemas``  — locate and load the JSON schemas under ``schemas/``.
- ``relsdlc.validate`` — validate records, edges, and cards; provenance,
  referential-integrity, and temporal-leakage gates.
- ``relsdlc.metrics``  — Recall@K, MRR, and hard-negative accuracy (pure stdlib).
- ``relsdlc.baseline`` — a dependency-light text embedding + retrieval baseline.
- ``relsdlc.cli``      — the ``relsdlc`` command-line entry point.
"""

__version__ = "0.1.0"
