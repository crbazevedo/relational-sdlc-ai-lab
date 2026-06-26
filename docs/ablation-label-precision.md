# Label-precision audit (R26) — the last `[PENDING]` item

**Status: exploratory, deterministic, committed-cache.** The benchmark calls its links
"verifiable." This audit quantifies that, separating two notions of precision honestly.
Script: [run_label_audit.py](../data/pilot/run_label_audit.py); result
[label-audit-results.json](../data/pilot/label-audit-results.json).

## Construction precision (did the mining rule fire correctly?)

| Relation | Construction precision | Definition |
|---|---|---|
| `fixes` (PR→issue) | **99.4%** (354/356) | the source PR's raw text contains a GitHub closing keyword (`close`/`fix`/`resolve` [+s/d]) referencing the **target issue number** |
| `modifies` (PR→file) | **100%** (1356/1356) | the edge correctly asserts "PR changed this file" (source is a PR, target a file/test node) — by git-history construction |

Construction precision is an **upper bound** on semantic precision.

A composition statistic, not a precision number: **29%** of `modifies` edges target a *test*
path (the rest are source files). The diff→test task uses that test subset; the 29% is the
graph's composition, not an error rate.

## Semantic precision (is the link meaningful?)

For `fixes` we report a **weak proxy** — token-Jaccard between the issue title and the PR
title — and a rendered 20-edge sample for manual rating (in the results JSON). The proxy:
median Jaccard 0.188, 88% of pairs have non-zero overlap.

The sample shows the **proxy badly understates** true semantic precision — the links are
correct even when lexical overlap is ~0:

| issue title | PR title | Jaccard | correct? |
|---|---|---|---|
| If the text contains `‍`, `console.print` may throw a RuntimeError | Fix cell_len crash when string ends with ZWJ | 0.00 | ✓ |
| `FileProxy.isatty()` always returns `False` instead of delegating | Fix `FileProxy.isatty()` always returning False instead of delegating | 0.71 | ✓ |
| Strings with ANSI escape sequences can cause `Console.print()` to hang | Fix infinite loop when ANSI escape sequences appear at string start | 0.15 | ✓ |

Every sampled pair is a genuine issue→fix match. So a lexical relatedness metric is a poor
gauge of semantic precision here, and — relevant to the wider paper — exactly the kind of
signal a text-similarity model would *miss*, which is why `fixes` is a useful supervision
label rather than something a regex/cosine already recovers.

## Verdict

The "verifiable relations" claim holds: `fixes` construction precision is 99.4% with an
independently-verifiable closing keyword (provenance method is tagged `human_label` for all
pilot edges, but the closing-keyword check is an *independent* confirmation), `modifies` is
100% by git history, and a manual sample confirms `fixes` is semantically correct even where
lexical overlap is absent. A larger human-rated study across both corpora would tighten the
semantic estimate, but the label foundation is sound. This was the final `[PENDING]`
empirical item; the sweep is complete.
