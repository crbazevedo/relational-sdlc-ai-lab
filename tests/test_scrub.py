"""Reference scrubbing removes pointers but keeps prose."""

from __future__ import annotations

from relsdlc.scrub import scrub_record_text, scrub_references


def test_removes_issue_numbers():
    assert "#123" not in scrub_references("This fixes #123 in the parser")
    assert "parser" in scrub_references("This fixes #123 in the parser")


def test_removes_gh_and_ownerrepo_refs():
    assert "gh-7" not in scrub_references("see gh-7 for context").lower()
    assert "#9" not in scrub_references("dup of owner/repo#9")


def test_removes_urls_and_shas():
    txt = "ref https://github.com/o/r/issues/42 and commit a1b2c3d4e5f6 done"
    out = scrub_references(txt)
    assert "github.com" not in out
    assert "a1b2c3d4e5f6" not in out
    assert "done" in out and "ref" in out


def test_keeps_plain_words_that_look_hexish():
    # 'deface' is letters-only (no digit) -> must survive the SHA rule.
    assert "deface" in scrub_references("do not deface the output")


def test_scrub_record_text_combines_title_body():
    rec = {"content": {"title": "Crash on #5", "body": "stack overflow in loop"}}
    out = scrub_record_text(rec)
    assert "#5" not in out
    assert "Crash" in out and "stack overflow" in out
