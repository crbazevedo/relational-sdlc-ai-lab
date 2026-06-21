"""Every schema is a valid JSON Schema and the cross-schema registry resolves."""

from __future__ import annotations

from jsonschema import Draft202012Validator

from relsdlc import schemas
from relsdlc.validate import _registry, _validator


def test_all_schemas_load():
    loaded = schemas.all_schemas()
    assert set(loaded) == set(schemas.SCHEMA_NAMES)


def test_schemas_are_valid_metaschema():
    for name in schemas.SCHEMA_NAMES:
        Draft202012Validator.check_schema(schemas.load_schema(name))


def test_registry_resolves_provenance_ref():
    # record + edge $ref provenance.schema.json; building a validator and using
    # it must not raise an unresolved-reference error.
    registry = _registry()
    assert registry is not None
    validator = _validator("record")
    # A minimally valid record should produce no errors.
    record = {
        "id": "r1",
        "type": "file",
        "provenance": {
            "source_url": "synthetic://x",
            "retrieved_at": "2024-01-01T00:00:00Z",
            "license": "CC0-1.0",
            "content_hash": "sha256:" + "0" * 64,
        },
    }
    assert list(validator.iter_errors(record)) == []
