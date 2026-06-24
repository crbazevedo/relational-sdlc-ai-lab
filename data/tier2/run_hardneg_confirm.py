#!/usr/bin/env python3
"""R19 — pin down the hardness lever: multi-seed confirmation + hard-negative mining.

R18 found, on a single seed, that same-repo "hard" batching beats a matched random
pool (ΔR@1 +0.137 vs +0.120). That +0.017 gap is ~2x MPS run-to-run noise, so this
wave (a) repeats it across SEEDS to see whether it clears the noise, and (b) adds a
stronger mechanism — explicit mined hard negatives — to test whether harder still
helps.

Three negative-hardness mechanisms, each at the SAME pool (batch 48), each at seeds
{0,1,2}, all vs the SAME frozen baseline (reused cache):

  - random    : in-batch negatives are whatever the random shuffle puts in the batch
  - repo-hard : repo-homogeneous batches (in-batch negs all from the anchor's repo)
  - mined     : random batches PLUS, per anchor, the same-repo non-gold PR with the
                highest issue/PR token overlap appended as an extra hard negative
                (forward InfoNCE over [positives ; mined-negs]; symmetric backward
                stays B x B)

Everything else matches R18 (MiniLM-L6 q/k/v LoRA r16/a32, symmetric InfoNCE temp
0.05 on TRAIN-repo pairs, 12 epochs, eval = raw cosine on held-out test repos).
Reuses run_negatives_sweep helpers; trained on the M5 GPU (MPS). Resumable.

Run: PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1 \
       .venv-np/bin/python data/tier2/run_hardneg_confirm.py > data/tier2/r19.log 2>&1
Smoke: RELSDLC_SMOKE=1 ...
"""
from __future__ import annotations

import gc
import json
import os
import statistics as st
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
from relsdlc.baseline import tokenize  # noqa: E402
from run_negatives_sweep import (  # noqa: E402  (reuse tested helpers)
    MODEL, EMB, MAX_LEN, TEMP, LR, _device, _embed, _embed_all, _batches,
)

RESULTS = HERE / "hardneg-confirm-results.json"
BATCH, EPOCHS = 48, 12
SEEDS = (0, 1)
# H1 (random vs repo-hard) first. mined (H2) is appended last and run only if the
# environment lets longer jobs finish; one cell per launch (RELSDLC_MAX_CELLS) keeps
# each run under the background-job time limit so it checkpoints cleanly.
CONFIGS = [("random", "random"), ("repo-hard", "repo"), ("mined", "mined")]
MAX_CELLS = int(os.environ.get("RELSDLC_MAX_CELLS", "99"))
SMOKE = os.environ.get("RELSDLC_SMOKE") == "1"
if SMOKE:
    SEEDS, EPOCHS = (0,), 1


def build_hardnegs(ds):
    """issue_id -> text of the same-repo, non-gold PR with the highest token overlap."""
    by = ds.by_id()
    gold = {iss: pr for pr, iss in ds.fixes}
    pr_by_repo: dict[str, list[tuple[str, set]]] = {}
    for a in ds.artifacts:
        if a.type == "pull_request" and a.split == "train":
            repo = a.id.split(":")[1]
            pr_by_repo.setdefault(repo, []).append((a.id, set(tokenize(a.text or ""))))
    out = {}
    for pr, iss in ds.fixes:
        if iss not in by or by[iss].split != "train":
            continue
        repo = iss.split(":")[1]
        itoks = set(tokenize(by[iss].text or ""))
        best, best_ov = None, -1
        for pid, ptoks in pr_by_repo.get(repo, []):
            if pid == gold.get(iss):
                continue
            ov = len(itoks & ptoks)
            if ov > best_ov:
                best_ov, best = ov, pid
        out[iss] = by[best].text if best else None
    return out


def train(seed, mode, ds, ids, texts, pairs, repos, hardneg, tok, torch, device):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModel

    t0 = time.time()
    torch.manual_seed(seed)
    model = get_peft_model(AutoModel.from_pretrained(MODEL), LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32,
        lora_dropout=0.05, target_modules=["query", "key", "value"])).to(device)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    ce = torch.nn.functional.cross_entropy
    rng = np.random.default_rng(seed)
    batch_mode = "repo" if mode == "repo" else "random"
    n = len(pairs)
    for ep in range(EPOCHS):
        tot = 0.0
        for idx in _batches(repos, BATCH, batch_mode, rng):
            if len(idx) < 2:
                continue
            a = _embed(model, tok, [pairs[j][0] for j in idx], torch, device, grad=True)
            b = _embed(model, tok, [pairs[j][1] for j in idx], torch, device, grad=True)
            lab = torch.arange(a.shape[0], device=device)
            if mode == "mined":
                hn = [hardneg[j] if hardneg[j] else pairs[(j + 1) % n][1] for j in idx]
                bn = _embed(model, tok, hn, torch, device, grad=True)
                ball = torch.cat([b, bn], 0)               # (2B, d)
                sims_f = (a @ ball.T) / TEMP               # B x 2B (extra hard negs)
                sims_b = (b @ a.T) / TEMP                  # B x B  (symmetric back)
                loss = (ce(sims_f, lab) + ce(sims_b, lab)) / 2
            else:
                sims = (a @ b.T) / TEMP
                loss = (ce(sims, lab) + ce(sims.T, lab)) / 2
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach())
        print(f"  [{mode} s{seed}] epoch {ep+1}/{EPOCHS} loss={tot:.3f}", file=sys.stderr, flush=True)
    model.eval()
    vecs = {str(i): v for i, v in zip(ids, _embed_all(model, tok, texts, torch, device))}
    m = run_cosine_on_vecs(ds, vecs)
    m["_minutes"] = round((time.time() - t0) / 60, 1)
    del model, opt
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return m


def main():
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        print(f"ERROR needs torch+transformers+peft: {exc}", file=sys.stderr); raise SystemExit(2)
    device = _device(torch)
    print(f"device={device.type} seeds={SEEDS} epochs={EPOCHS} smoke={SMOKE}", file=sys.stderr, flush=True)

    ds, meta = load_tier2_crossrepo()
    by = ds.by_id()
    ids = [a.id for a in ds.artifacts]
    texts = [a.text or a.id for a in ds.artifacts]
    hn_map = build_hardnegs(ds)
    pairs, repos, hardneg = [], [], []
    for pr, iss in ds.fixes:
        if iss in by and pr in by and by[iss].split == "train":
            pairs.append((by[iss].text or iss, by[pr].text or pr))
            repos.append(iss.split(":")[1] if ":" in iss else iss)
            hardneg.append(hn_map.get(iss))
    n_hn = sum(1 for h in hardneg if h)
    if SMOKE:
        pairs, repos, hardneg = pairs[:200], repos[:200], hardneg[:200]
    print(f"tier2: {len(pairs)} train pairs ({n_hn} with a mined hard neg), "
          f"{len(meta['test_repos'])} test repos", file=sys.stderr, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    fz = EMB / "minilm-l6-v2.npz"
    if fz.exists():
        d = np.load(fz, allow_pickle=False)
        V = np.asarray(d["vectors"], dtype=np.float32)
        frozen = run_cosine_on_vecs(ds, {str(i): V[k] for k, i in enumerate(d["ids"])})
        print("frozen cache reused", file=sys.stderr, flush=True)
    else:
        from transformers import AutoModel
        base = AutoModel.from_pretrained(MODEL).to(device); base.eval()
        frozen = run_cosine_on_vecs(ds, {str(i): v for i, v in zip(ids, _embed_all(base, tok, texts, torch, device))})
    fr1, frmrr = frozen["recall_at_k"]["1"], frozen["mrr"]
    print(f"RESULT frozen | R@1 {fr1:.3f} MRR {frmrr:.3f}", flush=True)

    cells = {}
    if not SMOKE and RESULTS.exists():
        cells = {c["key"]: c for c in json.loads(RESULTS.read_text()).get("cells", [])}
        print(f"resume: {len(cells)} cached -> {list(cells)}", file=sys.stderr, flush=True)

    def _write():
        rows = list(cells.values())
        agg = {}
        for name, _ in CONFIGS:
            ds1 = [c["delta_r1"] for c in rows if c["config"] == name]
            dmrr = [c["delta_mrr"] for c in rows if c["config"] == name]
            if ds1:
                agg[name] = {"n": len(ds1), "delta_r1_mean": round(st.mean(ds1), 4),
                             "delta_r1_std": round(st.pstdev(ds1), 4) if len(ds1) > 1 else 0.0,
                             "delta_mrr_mean": round(st.mean(dmrr), 4)}
        # paired deltas per seed (vs random)
        paired = {}
        for s in SEEDS:
            r = next((c["delta_r1"] for c in rows if c["config"] == "random" and c["seed"] == s), None)
            for name in ("repo-hard", "mined"):
                v = next((c["delta_r1"] for c in rows if c["config"] == name and c["seed"] == s), None)
                if r is not None and v is not None:
                    paired.setdefault(f"{name}_minus_random", []).append(round(v - r, 4))
        paired_summary = {k: {"per_seed": v, "mean": round(st.mean(v), 4),
                              "std": round(st.pstdev(v), 4) if len(v) > 1 else 0.0,
                              "all_positive": all(x > 0 for x in v)}
                          for k, v in paired.items()}
        RESULTS.write_text(json.dumps({
            "device": device.type, "batch": BATCH, "epochs": EPOCHS, "seeds": list(SEEDS),
            "frozen": {"recall_at_k": frozen["recall_at_k"], "mrr": frozen["mrr"]},
            "cells": rows, "aggregate": agg, "paired_vs_random": paired_summary,
            "meta": meta}, indent=2) + "\n", encoding="utf-8")

    ran = 0
    for name, mode in CONFIGS:           # config-outer: finish H1 (random/repo) across seeds before mined
        for seed in SEEDS:
            key = f"{name}-s{seed}"
            if key in cells:
                print(f"=== {key} (cached, skip) ===", file=sys.stderr, flush=True)
                continue
            print(f"=== {key} ===", file=sys.stderr, flush=True)
            try:
                m = train(seed, mode, ds, ids, texts, pairs, repos, hardneg, tok, torch, device)
            except Exception as exc:
                print(f"CELL FAILED {key}: {type(exc).__name__}: {exc}", flush=True)
                continue
            dr1 = m["recall_at_k"]["1"] - fr1
            dmrr = m["mrr"] - frmrr
            print(f"RESULT {key} | R@1 {m['recall_at_k']['1']:.3f} MRR {m['mrr']:.3f} "
                  f"| dR@1 {dr1:+.3f} dMRR {dmrr:+.3f} | {m['_minutes']}min", flush=True)
            cells[key] = {"key": key, "config": name, "seed": seed, "batch": BATCH,
                          "recall_at_k": m["recall_at_k"], "mrr": m["mrr"],
                          "delta_r1": round(dr1, 4), "delta_mrr": round(dmrr, 4),
                          "minutes": m["_minutes"]}
            if not SMOKE:
                _write()
            ran += 1
            if ran >= MAX_CELLS:
                print(f"MAX_CELLS={MAX_CELLS} reached — exiting cleanly (resumable)", flush=True)
                return
    print("R19 DONE" if not SMOKE else "SMOKE OK", flush=True)
    if SMOKE:
        for k, c in cells.items():
            print(f"  {k}: dR@1 {c['delta_r1']:+.3f}")


if __name__ == "__main__":
    main()
