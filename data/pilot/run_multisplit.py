#!/usr/bin/env python3
"""Track D — is the LoRA win robust across cross-repo splits, not seed/split luck?

For N different held-out-repo partitions, LoRA-fine-tune MiniLM on that split's
TRAIN-repo `fixes` pairs and evaluate frozen-cosine vs tuned-cosine on the split's
held-out test repos. Report the per-split deltas and their mean ± std.

Needs the [embed] extra (torch+transformers+peft). Writes a results JSON + a card;
the committed JSON is the snapshot a CI test reads (numpy only).

Run:  python data/pilot/run_multisplit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CARDS = REPO_ROOT / "data" / "cards" / "examples"
sys.path.insert(0, str(REPO_ROOT / "src"))

from relsdlc.metrics import RetrievalResult, evaluate  # noqa: E402
from relsdlc.scrub import scrub_record_text  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FROZEN = HERE / "embeddings" / "minilm-l6-v2.npz"
N_SPLITS = 5
TRAIN_FRAC = 0.6
EPOCHS = 10
BATCH = 32
LR = 2e-4
TEMP = 0.05
MAX_LEN = 256


def _load_jsonl(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def _repo_of(rid): return rid.split(":")[1] if len(rid.split(":")) > 1 else rid


def _frozen_vecs():
    d = np.load(FROZEN, allow_pickle=False)
    return {str(i): d["vectors"][k].astype(np.float32) for k, i in enumerate(d["ids"])}


def _eval(vecs, queries):
    res = []
    for q in queries:
        a = vecs[q["query_record"]]
        scored = sorted(q["candidates"], key=lambda c: (-float(a @ vecs[c]), c))
        res.append(RetrievalResult.of(scored, q["relevant"], q.get("hard_negatives", [])))
    return evaluate(res)


def _embed_all(model, tok, texts, torch):
    out = {}
    vecs = []
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        vecs.append(torch.nn.functional.normalize(e, p=2, dim=1).numpy())
    return np.concatenate(vecs, 0)


def main() -> None:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        print(f"ERROR: needs [embed] extra: {exc}", file=sys.stderr); raise SystemExit(2)

    records = _load_jsonl(HERE / "records.jsonl")
    text_of = {r["id"]: (scrub_record_text(r) or r["id"]) for r in records}
    ids = [r["id"] for r in records]
    queries_all = _load_jsonl(HERE / "benchmark" / "issue_to_fixing_pr.jsonl")
    repos = sorted({_repo_of(r["id"]) for r in records})
    frozen = _frozen_vecs()
    tok = AutoTokenizer.from_pretrained(MODEL)

    per_split = []
    for s in range(N_SPLITS):
        rng = np.random.default_rng(100 + s)
        perm = list(rng.permutation(len(repos)))
        train_repos = {repos[i] for i in perm[:int(len(repos) * TRAIN_FRAC)]}
        test_q = [q for q in queries_all if _repo_of(q["query_record"]) not in train_repos]
        pairs = [(text_of[q["query_record"]], text_of[q["relevant"][0]])
                 for q in queries_all if _repo_of(q["query_record"]) in train_repos]

        torch.manual_seed(0)
        base = AutoModel.from_pretrained(MODEL)
        model = get_peft_model(base, LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=8, lora_alpha=16,
            lora_dropout=0.05, target_modules=["query", "key", "value"]))
        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
        prng = np.random.default_rng(0)
        for _ in range(EPOCHS):
            order = prng.permutation(len(pairs))
            for i in range(0, len(pairs), BATCH):
                idx = order[i:i + BATCH]
                if len(idx) < 2:
                    continue
                q = [pairs[j][0] for j in idx]; d = [pairs[j][1] for j in idx]
                ea = _embed_grad(model, tok, q, torch); eb = _embed_grad(model, tok, d, torch)
                sims = (ea @ eb.T) / TEMP
                lab = torch.arange(ea.shape[0])
                loss = (torch.nn.functional.cross_entropy(sims, lab)
                        + torch.nn.functional.cross_entropy(sims.T, lab)) / 2
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        tuned_mat = _embed_all(model, tok, [text_of[i] for i in ids], torch)
        tuned = {i: tuned_mat[k] for k, i in enumerate(ids)}

        fz, tu = _eval(frozen, test_q), _eval(tuned, test_q)
        row = {"split": s, "n_test_repos": len(repos) - len(train_repos),
               "n_test_q": len(test_q),
               "frozen_r1": fz["recall_at_k"]["1"], "tuned_r1": tu["recall_at_k"]["1"],
               "frozen_mrr": fz["mrr"], "tuned_mrr": tu["mrr"],
               "delta_r1": tu["recall_at_k"]["1"] - fz["recall_at_k"]["1"],
               "delta_mrr": tu["mrr"] - fz["mrr"]}
        per_split.append(row)
        print(f"split {s}: frozen R@1={row['frozen_r1']:.3f} tuned R@1={row['tuned_r1']:.3f} "
              f"(Δ={row['delta_r1']:+.3f})  frozen MRR={row['frozen_mrr']:.3f} "
              f"tuned MRR={row['tuned_mrr']:.3f} (Δ={row['delta_mrr']:+.3f})", file=sys.stderr)

    def ms(key):
        a = np.array([r[key] for r in per_split]); return float(a.mean()), float(a.std())
    agg = {k: {"mean": ms(k)[0], "std": ms(k)[1]}
           for k in ["frozen_r1", "tuned_r1", "delta_r1", "frozen_mrr", "tuned_mrr", "delta_mrr"]}
    out = {"model": MODEL + "+lora", "n_splits": N_SPLITS, "epochs": EPOCHS,
           "per_split": per_split, "aggregate": agg}
    (HERE / "multisplit-results.json").write_text(json.dumps(out, indent=2) + "\n")

    print("\n=== AGGREGATE over %d cross-repo splits ===" % N_SPLITS)
    print(f"frozen  R@1 {agg['frozen_r1']['mean']:.3f} ± {agg['frozen_r1']['std']:.3f} | "
          f"MRR {agg['frozen_mrr']['mean']:.3f} ± {agg['frozen_mrr']['std']:.3f}")
    print(f"LoRA    R@1 {agg['tuned_r1']['mean']:.3f} ± {agg['tuned_r1']['std']:.3f} | "
          f"MRR {agg['tuned_mrr']['mean']:.3f} ± {agg['tuned_mrr']['std']:.3f}")
    print(f"delta   R@1 {agg['delta_r1']['mean']:+.3f} ± {agg['delta_r1']['std']:.3f} | "
          f"MRR {agg['delta_mrr']['mean']:+.3f} ± {agg['delta_mrr']['std']:.3f}")


def _embed_grad(model, tok, texts, torch):
    enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    h = model(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).float()
    e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(e, p=2, dim=1)


if __name__ == "__main__":
    main()
