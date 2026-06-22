#!/usr/bin/env python3
"""GraphRAG → small-SLM **dry-run** demo (MVP-2 entry, Track E).

This is the second, best-effort half of the relational-SLM layer the
[architecture](../../docs/architecture.md) sketches:

    task -> relation-conditioned retrieval -> subgraph -> prompt/context -> response

The first half — the deterministic, CI-tested **subgraph packer**
([`src/relsdlc/subgraph.py`](../../src/relsdlc/subgraph.py)) — is the real
deliverable. This script feeds the packed GraphRAG context to a *small instruct
model* and asks it for a short structured triage (likely-fix summary, which files
to look at, suggested tests). It is a **DRY-RUN demonstration, not a benchmark**:
greedy decoding, CPU, no quantitative claim, no scoring. The point is to show the
end-to-end shape — relation-packed context in, structured triage out.

Model: ``Qwen/Qwen2.5-0.5B-Instruct`` (~0.5B params, the smallest viable
instruct model; ~1 GB download). CPU, greedy, short ``max_new_tokens``.

Run (needs the embed extra: ``pip install -e '.[embed]'``):

    python data/pilot/slm_demo.py

Output: ``data/pilot/slm-outputs/*.md`` — the packed context + the model's
generation for each sample issue (committed samples). If the model download or
CPU generation is infeasible, the packer still runs and writes the packed
contexts to ``slm-outputs/`` with a clear "generation deferred" note — the
retrieval half is the deliverable that always ships.

This script is NOT in CI (it needs torch + transformers + a one-time download).
It is run once locally to produce the committed sample outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from relsdlc.subgraph import build_graph_view, pack_context, retrieve_subgraph  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT_DIR = HERE / "slm-outputs"
TOP_K = 5
MAX_NEW_TOKENS = 160

# Three fixed sample issues from the pilot (Textualize/rich) — chosen because
# their top-k retrieval includes a PR with concrete changed files/tests, so the
# packed context is non-trivial. Deterministic, no random sampling.
SAMPLE_ISSUES = [
    "gh:Textualize/rich:issue:3577",
    "gh:Textualize/rich:issue:3958",
    "gh:Textualize/rich:issue:3947",
]

SYSTEM_PROMPT = (
    "You are an SDLC triage assistant. You are given a software issue and the "
    "most relevant pull requests retrieved from a code-change graph, each with "
    "the source files and tests it changed. Produce a SHORT structured triage:\n"
    "1. Likely fix (one sentence).\n"
    "2. Files to look at (bullet list of paths).\n"
    "3. Suggested tests (bullet list of paths).\n"
    "Base your answer ONLY on the provided context. Be concise."
)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _build_graph() -> object:
    records = _load_jsonl(HERE / "records.jsonl")
    file_records = _load_jsonl(HERE / "graph" / "file_records.jsonl")
    fixes = [
        (e["source"], e["target"])
        for e in _load_jsonl(HERE / "edges.jsonl")
        if e.get("relation") == "fixes"
    ]
    mods = [
        (e["source"], e["target"])
        for e in _load_jsonl(HERE / "graph" / "modifies_edges.jsonl")
        if e.get("relation") == "modifies"
    ]
    npz = np.load(HERE / "embeddings" / "minilm-l6-v2.npz", allow_pickle=True)
    ids = [str(i) for i in npz["ids"]]
    vecs = np.asarray(npz["vectors"], dtype=np.float64)
    node_vecs = {i: vecs[k] for k, i in enumerate(ids)}
    return build_graph_view(records, file_records, fixes, mods, node_vecs)


def _try_load_model():
    """Return ``(tokenizer, model)`` or ``None`` if the embed extra is unavailable."""
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        print(f"NOTE: embed extra unavailable ({exc}); generation deferred.", file=sys.stderr)
        return None
    try:
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto")
        model.eval()
    except Exception as exc:  # pragma: no cover - network / download failure
        print(f"NOTE: could not load {MODEL} ({exc}); generation deferred.", file=sys.stderr)
        return None
    return tok, model


def _generate(tok, model, context: str) -> str:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,           # greedy, deterministic
            num_beams=1,
            pad_token_id=tok.eos_token_id,
        )
    gen = out[0][enc["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def _write_output(issue_id: str, context: str, generation: str | None) -> Path:
    safe = issue_id.replace(":", "_").replace("/", "_")
    path = OUT_DIR / f"{safe}.md"
    lines = [
        f"# SLM dry-run — `{issue_id}`",
        "",
        "**This is a DRY-RUN demo (MVP-2 entry), not a benchmark.** No quantitative "
        "claim is made about the generation quality; the deterministic GraphRAG "
        "packer in `src/relsdlc/subgraph.py` is the evaluated deliverable.",
        "",
        f"- Model: `{MODEL}` (CPU, greedy, max_new_tokens={MAX_NEW_TOKENS})",
        f"- Retrieval: relation-conditioned, top_k={TOP_K} (cosine + `modifies` expansion)",
        "",
        "## Packed GraphRAG context (input to the SLM)",
        "",
        "```text",
        context.rstrip(),
        "```",
        "",
        "## SLM triage (generation)",
        "",
    ]
    if generation is None:
        lines += [
            "_Generation deferred — the `[embed]` extra (transformers + torch) or the "
            "model download was unavailable in this environment. The packed context "
            "above is the real, deterministic deliverable; re-run "
            "`python data/pilot/slm_demo.py` after `pip install -e '.[embed]'` to fill "
            "this section._",
        ]
    else:
        lines += ["```text", generation, "```"]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_readme(deferred: bool) -> None:
    status = (
        "Generation was DEFERRED in the environment that produced this snapshot "
        "(the `[embed]` extra or the model download was unavailable). The packed "
        "GraphRAG contexts below are the real, deterministic deliverable; the SLM "
        "generation can be filled in by re-running the demo after "
        "`pip install -e '.[embed]'`."
        if deferred
        else
        f"Generations were produced once locally with `{MODEL}` on CPU (greedy "
        "decoding). They are committed samples of a DRY-RUN demo — there is no "
        "benchmark or quantitative claim attached to them."
    )
    readme = OUT_DIR / "README.md"
    readme.write_text(
        "# SLM dry-run outputs (MVP-2, Track E)\n\n"
        "Committed samples of the **GraphRAG → small-SLM dry-run** "
        "(`data/pilot/slm_demo.py`). The deterministic subgraph packer "
        "(`src/relsdlc/subgraph.py`) is the CI-tested deliverable; this directory "
        "holds best-effort generation samples only.\n\n"
        f"**Status.** {status}\n\n"
        "Each `gh_*.md` file shows, for one sample issue: the relation-conditioned "
        "packed context (issue + top-k related PRs with their changed files/tests) "
        "and the SLM's structured triage (or a deferral note).\n\n"
        "See [`docs/slm-dryrun.md`](../../../docs/slm-dryrun.md) for the write-up.\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    graph = _build_graph()
    loaded = _try_load_model()
    tok = model = None
    if loaded is not None:
        tok, model = loaded

    deferred = loaded is None
    for issue_id in SAMPLE_ISSUES:
        subgraph = retrieve_subgraph(issue_id, graph, top_k=TOP_K)
        context = pack_context(subgraph)
        generation = None
        if loaded is not None:
            print(f"generating triage for {issue_id} ...", file=sys.stderr)
            try:
                generation = _generate(tok, model, context)
            except Exception as exc:  # pragma: no cover
                print(f"NOTE: generation failed for {issue_id} ({exc}).", file=sys.stderr)
                generation = None
                deferred = True
        path = _write_output(issue_id, context, generation)
        print(f"wrote {path}")

    _write_readme(deferred)
    print(f"wrote {OUT_DIR / 'README.md'}")
    if deferred:
        print("generation deferred — packed contexts written; see slm-outputs/README.md")


if __name__ == "__main__":
    main()
