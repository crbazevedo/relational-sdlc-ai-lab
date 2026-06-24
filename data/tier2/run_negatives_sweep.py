#!/usr/bin/env python3
"""R18 — push R16D's negative-pool lever on the Apple M5 GPU (PyTorch MPS).

R16D found the Tier-2 LoRA win rank-saturated, with harder/more in-batch negatives
the only lever, stalled at in-batch pool 48 by CPU memory. This trains on the M5 GPU
(MPS), where 32 GB unified memory lifts the cap, and sweeps:

  H1 (quantity): batch in {32, 48, 96, 192, 384} = the in-batch negative pool,
                 random batching. (b32/b48 reproduce R16C/R16D as anchors.)
  H2 (hardness): batch 48 with REPO-HOMOGENEOUS batches (in-batch negatives all from
                 the anchor's own repo -> semantically harder) vs random at the same
                 size.

Everything else matches finetune_tier2.py / sweep_tier2_lora.py exactly (MiniLM-L6
q/k/v LoRA, symmetric InfoNCE on TRAIN-repo pairs only, 12 epochs, eval = raw cosine
on HELD-OUT test repos, references scrubbed). Saves frozen + best-LoRA embeddings to
data/tier2/embeddings/ (gitignored) for the R18 bootstrap CI step.

Run (live log, sleep-proof):
  PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1 \
    .venv-np/bin/python data/tier2/run_negatives_sweep.py > data/tier2/r18-sweep.log 2>&1
Smoke (1 epoch, one cell, subset):  RELSDLC_SMOKE=1 ... run_negatives_sweep.py
"""
from __future__ import annotations

import gc
import json
import os
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
RESULTS = HERE / "negatives-sweep-results.json"
SEED, EPOCHS, LR, TEMP, MAX_LEN, EMBED_CHUNK = 0, 12, 2e-4, 0.05, 256, 64
SMOKE = os.environ.get("RELSDLC_SMOKE") == "1"

# (name, rank, batch, batching). r16-b32 reproduces the shipped baseline (ΔR@1 +0.114).
CONFIGS = [
    ("r16-b32 (R16C anchor)", 16, 32, "random"),
    ("r16-b48 (R16D best)", 16, 48, "random"),
    ("r16-b96", 16, 96, "random"),
    # b192 / b384 OOM the M5 MPS allocator (InfoNCE backprop over 384/768 texts needs
    # >42 GiB) — the practical in-batch pool ceiling on this machine is ~b96.
    ("r16-b48 repo-hard", 16, 48, "repo"),  # H2: matched pool, harder negatives
]
if SMOKE:
    EPOCHS = 1
    CONFIGS = [("r16-b32 SMOKE", 16, 32, "random")]


def _device(torch):
    forced = os.environ.get("RELSDLC_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _embed(model, tok, texts, torch, device, grad=False):
    enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        h = model(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).float()
    e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(e, p=2, dim=1)


def _embed_all(model, tok, texts, torch, device):
    out = []
    for i in range(0, len(texts), EMBED_CHUNK):
        out.append(_embed(model, tok, texts[i:i + EMBED_CHUNK], torch, device).detach().cpu().numpy())
    return np.concatenate(out, 0).astype(np.float32)


def _batches(repos, batch, mode, rng):
    """Yield arrays of pair indices. mode='random' shuffles all; mode='repo' groups by
    repository so in-batch negatives are same-repo (harder)."""
    n = len(repos)
    if mode == "random":
        order = rng.permutation(n)
    else:  # repo-homogeneous: shuffle repo order + within-repo, then concatenate
        by_repo: dict[str, list[int]] = {}
        for i, rp in enumerate(repos):
            by_repo.setdefault(rp, []).append(i)
        keys = list(by_repo)
        rng.shuffle(keys)
        order = []
        for k in keys:
            idx = by_repo[k]
            rng.shuffle(idx)
            order.extend(idx)
        order = np.array(order)
    for i in range(0, n, batch):
        yield order[i:i + batch]


def train_and_eval(name, r, batch, mode, ds, ids, texts, pairs, repos, tok, torch, device):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModel

    t0 = time.time()
    model = get_peft_model(AutoModel.from_pretrained(MODEL), LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=r, lora_alpha=2 * r,
        lora_dropout=0.05, target_modules=["query", "key", "value"])).to(device)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    ce = torch.nn.functional.cross_entropy
    rng = np.random.default_rng(SEED)
    for ep in range(EPOCHS):
        tot = 0.0
        for idx in _batches(repos, batch, mode, rng):
            if len(idx) < 2:
                continue
            a = _embed(model, tok, [pairs[j][0] for j in idx], torch, device, grad=True)
            b = _embed(model, tok, [pairs[j][1] for j in idx], torch, device, grad=True)
            sims = (a @ b.T) / TEMP
            lab = torch.arange(a.shape[0], device=device)
            loss = (ce(sims, lab) + ce(sims.T, lab)) / 2
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach())
        print(f"  [{name}] epoch {ep + 1}/{EPOCHS} loss={tot:.3f}", file=sys.stderr, flush=True)
    model.eval()
    vecs = {str(i): v for i, v in zip(ids, _embed_all(model, tok, texts, torch, device))}
    m = run_cosine_on_vecs(ds, vecs)
    m["_minutes"] = round((time.time() - t0) / 60, 1)
    # Free the GPU allocator between cells — without this, MPS memory accumulates
    # across configs and per-cell wall-time creeps up (7min -> 18min observed).
    del model, opt
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return m, vecs


def main() -> None:
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        print(f"ERROR: needs torch+transformers+peft: {exc}", file=sys.stderr); raise SystemExit(2)
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = _device(torch)
    EMB.mkdir(parents=True, exist_ok=True)
    print(f"device={device.type}  epochs={EPOCHS}  smoke={SMOKE}", file=sys.stderr, flush=True)

    ds, meta = load_tier2_crossrepo()
    by = ds.by_id()
    ids = [a.id for a in ds.artifacts]
    texts = [a.text or a.id for a in ds.artifacts]
    pairs, repos = [], []
    for pr, iss in ds.fixes:
        if iss in by and pr in by and by[iss].split == "train":
            pairs.append((by[iss].text or iss, by[pr].text or pr))
            repos.append(iss.split(":")[1] if ":" in iss else iss)  # owner/repo
    if SMOKE:
        pairs, repos = pairs[:200], repos[:200]
    print(f"tier2: {len(ids)} records, {len(pairs)} train pairs, "
          f"{len(meta['train_repos'])} train / {len(meta['test_repos'])} test repos",
          file=sys.stderr, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)

    # Frozen baseline (anchor: reproduces R16C R@1 ~0.515). Reuse a prior frozen cache
    # if present (deterministic) so a resume is cheap.
    fz = EMB / "minilm-l6-v2.npz"
    if fz.exists():
        d = np.load(fz, allow_pickle=False)
        V = np.asarray(d["vectors"], dtype=np.float32)
        frozen_vecs = {str(i): V[k] for k, i in enumerate(d["ids"])}
        print("frozen cache present — reusing", file=sys.stderr, flush=True)
    else:
        from transformers import AutoModel
        base = AutoModel.from_pretrained(MODEL).to(device); base.eval()
        frozen_vecs = {str(i): v for i, v in zip(ids, _embed_all(base, tok, texts, torch, device))}
        if not SMOKE:
            np.savez_compressed(fz, ids=np.array(ids),
                                vectors=np.stack([frozen_vecs[str(i)] for i in ids]).astype(np.float16),
                                model=MODEL)
    frozen = run_cosine_on_vecs(ds, frozen_vecs)
    print(f"RESULT frozen | R@1 {frozen['recall_at_k']['1']:.3f} R@5 {frozen['recall_at_k']['5']:.3f} "
          f"MRR {frozen['mrr']:.3f} HNA {frozen['hard_negative_accuracy']:.3f}", flush=True)

    # Resume support: carry forward any cells a prior (killed) run already wrote.
    done = {}
    if not SMOKE and RESULTS.exists():
        for row in json.loads(RESULTS.read_text()).get("configs", []):
            done[row["name"]] = row
        print(f"resume: {len(done)} cell(s) cached -> {list(done)}", file=sys.stderr, flush=True)

    rows = [done[n] for n in done]
    best = {"delta_r1": max([r.get("delta_r1", -1.0) for r in rows
                             if r.get("batching") == "random"] + [-1.0])}

    def _write():
        out = {"device": device.type, "epochs": EPOCHS, "max_len": MAX_LEN, "seed": SEED,
               "model": MODEL, "lora": {"r": 16, "alpha": 32, "targets": ["query", "key", "value"]},
               "temp": TEMP,
               "frozen": {k: frozen[k] for k in ("recall_at_k", "mrr", "hard_negative_accuracy", "n_queries")},
               "best_random_cell": max((r for r in rows if r.get("batching") == "random"),
                                       key=lambda r: r.get("delta_r1", -1), default={}).get("name"),
               "configs": rows, "meta": meta}
        RESULTS.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    for k, (name, r, batch, mode) in enumerate(CONFIGS, 1):
        if name in done:
            print(f"=== CONFIG {k}/{len(CONFIGS)}: {name} (cached, skip) ===", file=sys.stderr, flush=True)
            continue
        print(f"=== CONFIG {k}/{len(CONFIGS)}: {name} (r={r}, batch={batch}, {mode}) ===",
              file=sys.stderr, flush=True)
        try:
            m, vecs = train_and_eval(name, r, batch, mode, ds, ids, texts, pairs, repos, tok, torch, device)
        except Exception as exc:  # keep the sweep alive on a single-config failure
            print(f"CONFIG FAILED {name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        dr1 = m["recall_at_k"]["1"] - frozen["recall_at_k"]["1"]
        dmrr = m["mrr"] - frozen["mrr"]
        print(f"RESULT {name} | R@1 {m['recall_at_k']['1']:.3f} R@5 {m['recall_at_k']['5']:.3f} "
              f"MRR {m['mrr']:.3f} HNA {m['hard_negative_accuracy']:.3f} "
              f"| dR@1 {dr1:+.3f} dMRR {dmrr:+.3f} | {m['_minutes']}min", flush=True)
        rows.append({"name": name, "r": r, "batch": batch, "batching": mode,
                     "delta_r1": round(dr1, 4), "delta_mrr": round(dmrr, 4),
                     "recall_at_k": m["recall_at_k"], "mrr": m["mrr"],
                     "hard_negative_accuracy": m["hard_negative_accuracy"],
                     "n_queries": m.get("n_queries", 0), "minutes": m["_minutes"]})
        if not SMOKE:
            _write()  # incremental: a kill keeps every finished cell
            if dr1 > best["delta_r1"]:
                best["delta_r1"] = dr1
                np.savez_compressed(EMB / "minilm-lora-best.npz", ids=np.array(ids),
                                    vectors=np.stack([vecs[str(i)] for i in ids]).astype(np.float16),
                                    model=MODEL + "+lora", config=name)
                print(f"  new best random cell ({name}) -> saved minilm-lora-best.npz", flush=True)

    print(f"SWEEP DONE -> {RESULTS}" if not SMOKE else "SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
