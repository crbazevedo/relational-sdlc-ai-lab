# SLM dry-run — `gh:Textualize/rich:issue:3577`

**This is a DRY-RUN demo (MVP-2 entry), not a benchmark.** No quantitative claim is made about the generation quality; the deterministic GraphRAG packer in `src/relsdlc/subgraph.py` is the evaluated deliverable.

- Model: `Qwen/Qwen2.5-0.5B-Instruct` (CPU, greedy, max_new_tokens=160)
- Retrieval: relation-conditioned, top_k=5 (cosine + `modifies` expansion)

## Packed GraphRAG context (input to the SLM)

```text
# ISSUE
title: [BUG] Trailing line break removed by Text.from_ansi().
description: - [x] I've checked [docs](https://rich.readthedocs.io/en/latest/introduction.html) and [closed issues](https://github.com/Textualize/rich/issues?q=is%3Aissue+is%3Aclosed) for possible solutions. - [x] I can't find my issue in the [FAQ](https://github.com/Textualize/rich/blob/master/FAQ.md). **Describe the bug** If the string passed to `Text.from_ansi()` ends with a newline, it is removed. Simply put : `Text("\n") != Text.from_ansi("\n")` **Example** ```Python from rich.text import Text prin

# RELATED PULL REQUESTS (top 5 by relevance)
## PR #1  (relevance=0.714)
summary: Fixed Text.from_ansi() removing trailing line break.
changed files: (none recorded)
changed tests: (none recorded)

## PR #2  (relevance=0.667)
summary: Fix Text.from_ansi() stripping trailing newlines
changed files: rich/text.py
changed tests: tests/test_ansi.py

## PR #3  (relevance=0.616)
summary: Disable html_inline in Markdown to prevent text swallowing
changed files: (none recorded)
changed tests: (none recorded)

## PR #4  (relevance=0.591)
summary: Fix infinite loop when ANSI escape sequences appear at string start
changed files: rich/cells.py
changed tests: tests/test_cells.py

## PR #5  (relevance=0.580)
summary: fix: additional newline added to docstring when the previous line length is l
changed files: (none recorded)
changed tests: (none recorded)
```

## SLM triage (generation)

```text
### Likely fix
- The issue arises because `Text.from_ansi()` removes trailing newline characters if they exist.

### Files to Look At
- `rich/text.py`
- `tests/test_ansi.py`

### Suggested Tests
- `tests/test_ansi.py`: Verify that `Text.from_ansi()` correctly handles trailing newlines.
- `rich/cells.py`: Ensure that `Text.from_ansi()` does not remove trailing newlines unintentionally.
```
