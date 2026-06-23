#!/usr/bin/env python3
"""LoRA hyperparameter sweep on the dense Tier-2 set: rank x harder-negatives.

Answers one question: is the shipped r=8 / batch=32 result (ΔR@1 +0.114) a FLOOR
or already near-tuned? We vary two axes against the SAME fixed frozen baseline:

  - rank: r in {8, 16, 32} at batch 32 (LoRA capacity; adapters are tiny, so
    memory is ~constant)
  - harder negatives: in-batch InfoNCE pool 32 vs 48 at r=16 (a larger pool means
    more — and on average harder — negatives per anchor; SimCLR/InfoNCE effect)

Everything else matches finetune_tier2.py exactly (MiniLM-L6 q/k/v, symmetric
InfoNCE on TRAIN-repo fixes pairs only, 12 epochs, eval = raw cosine on HELD-OUT
test repos with references scrubbed). Embeddings are computed in-memory (no npz
round-trip). CPU-only training; numpy eval. Each config is independently guarded
so one failure (e.g. OOM) does not abort the rest of the sweep.

Run (memory-safe, sleep-proof, live log):
  caffeinate -ims env PYTHONUNBUFFERED=1 .venv-embed/bin/python \
    data/tier2/sweep_tier2_lora.py > data/tier2/sweep.log 2>&1
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from run_tier2_ablation import load_tier2_crossrepo  # noqa: E402
from relsdlc.tower import run_cosine_on_vecs  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMB = HERE / "embeddings"
SEED, EPOCHS, LR, TEMP, MAX_LEN, EMBED_CHUNK = 0, 12, 2e-4, 0.05, 256, 32

# (name, rank, batch). target_modules fixed to q/k/v to keep this a clean
# rank x in-batch-negatives sweep. r8-b32 reproduces the shipped baseline.
CONFIGS = [
    ("r8-b32 (baseline)", 8, 32),
    ("r16-b32", 16, 32),
    ("r32-b32", 32, 32),
    ("r16-b48 (harder negs)", 16, 48),
]
TARGET_MODULES = ["query", "key", "value"]


def _embed(model, tok, texts, torch, grad=False):
    enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        h = model(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).float()
    e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(e, p=2, dim=1)


def _embed_all(model, tok, texts, torch):
    out = []
    for i in range(0, len(texts), EMBED_CHUNK):
        out.append(_embed(model, tok, texts[i:i + EMBED_CHUNK], torch).numpy())
    return np.concatenate(out, 0).astype(np.float32)


def train_and_eval(name, r, batch, ds, ids, texts, pairs, tok, torch):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModel

    t0 = time.time()
    model = get_peft_model(AutoModel.from_pretrained(MODEL), LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=r, lora_alpha=2 * r,
        lora_dropout=0.05, target_modules=TARGET_MODULES))
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    rng = np.random.default_rng(SEED)
    ce = torch.nn.functional.cross_entropy
    for ep in range(EPOCHS):
        order = rng.permutation(len(pairs)); tot = 0.0
        for i in range(0, len(pairs), batch):
            idx = order[i:i + batch]
            if len(idx) < 2:
                continue
            a = _embed(model, tok, [pairs[j][0] for j in idx], torch, grad=True)
            b = _embed(model, tok, [pairs[j][1] for j in idx], torch, grad=True)
            sims = (a @ b.T) / TEMP
            lab = torch.arange(a.shape[0])
            loss = (ce(sims, lab) + ce(sims.T, lab)) / 2
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        print(f"  [{name}] epoch {ep + 1}/{EPOCHS} loss={tot:.3f}", file=sys.stderr, flush=True)
    model.eval()
    vecs = {str(i): v for i, v in zip(ids, _embed_all(model, tok, texts, torch))}
    m = run_cosine_on_vecs(ds, vecs)
    m["_minutes"] = round((time.time() - t0) / 60, 1)
    return m


def main() -> None:
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        print(f"ERROR: needs [embed] extra: {exc}", file=sys.stderr); raise SystemExit(2)
    torch.manual_seed(SEED); np.random.seed(SEED)

    ds, meta = load_tier2_crossrepo()
    by = ds.by_id()
    ids = [a.id for a in ds.artifacts]
    texts = [a.text or a.id for a in ds.artifacts]
    pairs = [(by[iss].text or iss, by[pr].text or pr) for pr, iss in ds.fixes
             if iss in by and pr in by and by[iss].split == "train"]
    print(f"tier2 sweep: {len(ids)} records, {len(pairs)} train pairs, "
          f"{len(meta['train_repos'])} train / {len(meta['test_repos'])} test repos",
          file=sys.stderr, flush=True)

    # Fixed frozen baseline (reuse the committed-recipe frozen cache if present).
    tok = AutoTokenizer.from_pretrained(MODEL)
    if (EMB / "minilm-l6-v2.npz").exists():
        d = np.load(EMB / "minilm-l6-v2.npz", allow_pickle=False)
        fids, V = d["ids"], np.asarray(d["vectors"], dtype=np.float32)
        frozen_vecs = {str(i): V[k] for k, i in enumerate(fids)}
    else:
        from transformers import AutoModel
        base = AutoModel.from_pretrained(MODEL); base.eval()
        frozen_vecs = {str(i): v for i, v in zip(ids, _embed_all(base, tok, texts, torch))}
    frozen = run_cosine_on_vecs(ds, frozen_vecs)
    print(f"RESULT frozen | R@1 {frozen['recall_at_k']['1']:.3f} R@5 {frozen['recall_at_k']['5']:.3f} "
          f"R@10 {frozen['recall_at_k']['10']:.3f} MRR {frozen['mrr']:.3f} "
          f"HNA {frozen['hard_negative_accuracy']:.3f}", flush=True)

    rows = [{"name": "frozen", **frozen}]
    for k, (name, r, batch) in enumerate(CONFIGS, 1):
        print(f"=== CONFIG {k}/{len(CONFIGS)}: {name} (r={r}, batch={batch}) ===",
              file=sys.stderr, flush=True)
        try:
            m = train_and_eval(name, r, batch, ds, ids, texts, pairs, tok, torch)
        except Exception as exc:  # keep the sweep alive on a single-config failure
            print(f"CONFIG FAILED {name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        dr1 = m["recall_at_k"]["1"] - frozen["recall_at_k"]["1"]
        dmrr = m["mrr"] - frozen["mrr"]
        print(f"RESULT {name} | R@1 {m['recall_at_k']['1']:.3f} R@5 {m['recall_at_k']['5']:.3f} "
              f"R@10 {m['recall_at_k']['10']:.3f} MRR {m['mrr']:.3f} HNA {m['hard_negative_accuracy']:.3f} "
              f"| dR@1 {dr1:+.3f} dMRR {dmrr:+.3f} | {m['_minutes']}min", flush=True)
        rows.append({"name": name, "r": r, "batch": batch, "delta_r1": dr1,
                     "delta_mrr": dmrr, **m})

    (HERE / "tier2-sweep-results.json").write_text(
        json.dumps({"frozen": frozen, "configs": rows, "meta": meta,
                    "epochs": EPOCHS, "max_len": MAX_LEN}, indent=2) + "\n", encoding="utf-8")
    print("SWEEP DONE -> data/tier2/tier2-sweep-results.json", flush=True)


if __name__ == "__main__":
    main()
