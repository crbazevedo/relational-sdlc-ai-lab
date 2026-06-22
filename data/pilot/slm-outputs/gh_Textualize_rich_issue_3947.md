# SLM dry-run — `gh:Textualize/rich:issue:3947`

**This is a DRY-RUN demo (MVP-2 entry), not a benchmark.** No quantitative claim is made about the generation quality; the deterministic GraphRAG packer in `src/relsdlc/subgraph.py` is the evaluated deliverable.

- Model: `Qwen/Qwen2.5-0.5B-Instruct` (CPU, greedy, max_new_tokens=160)
- Retrieval: relation-conditioned, top_k=5 (cosine + `modifies` expansion)

## Packed GraphRAG context (input to the SLM)

```text
# ISSUE
title: [BUG] If the text contains '\u200d', console.print may throw a RuntimeError
description: - [x] I've checked [docs](https://rich.readthedocs.io/en/latest/introduction.html) and [closed issues](https://github.com/Textualize/rich/issues?q=is%3Aissue+is%3Aclosed) for possible solutions. - [x] I can't find my issue in the [FAQ](https://github.com/Textualize/rich/blob/master/FAQ.md). <!-- WAIT! Is this a feature request? Rich isn't accepting new features. If it can be implemented in a third-party library, then that is what I recommend. Rich *may* accept enhancements to existing featur

# RELATED PULL REQUESTS (top 5 by relevance)
## PR #1  (relevance=0.670)
summary: fix(typo): correct typo in Text.divide docstring
changed files: (none recorded)
changed tests: (none recorded)

## PR #2  (relevance=0.663)
summary: Fixed Text.from_ansi() removing trailing line break.
changed files: (none recorded)
changed tests: (none recorded)

## PR #3  (relevance=0.622)
summary: Disable html_inline in Markdown to prevent text swallowing
changed files: (none recorded)
changed tests: (none recorded)

## PR #4  (relevance=0.620)
summary: 🐛 Fix `TYPER_USE_RICH` parsing to allow disabling Rich completely
changed files: (none recorded)
changed tests: (none recorded)

## PR #5  (relevance=0.615)
summary: Fix typing for save_html, save_text, save_svg to accept PathLike
changed files: (none recorded)
changed tests: (none recorded)
```

## SLM triage (generation)

```text
### Likely fix
The issue is related to the `console.print` function throwing a `RuntimeError` when encountering `\u200d`. This suggests there might be a bug in how the string is being processed.

### Files to Look At
- `text.py`
- `text.pyx`

### Suggested Tests
- `test_rich.py`
- `test_rich.pyx`
```
