#!/usr/bin/env python3
"""LoRA-at-scale: frozen + LoRA-tuned MiniLM embeddings of the ~55-repo dataset.

Confirms (or refutes) that the pilot's LoRA win scales. Same recipe as the pilot
(`data/pilot/finetune_embed.py`): LoRA r=8 on attention q/k/v, symmetric InfoNCE on
TRAIN-repo `fixes` pairs only, evaluated on held-out test repos. Caches both frozen
and tuned embeddings (committed); eval is numpy-only (`run_scale_finetune.py`).

Needs the [embed] extra. Run:  python data/scale/finetune_scale.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from run_scale_ablation import load_scale_crossrepo  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMB = HERE / "embeddings"
SEED, EPOCHS, BATCH, LR, TEMP, MAX_LEN = 0, 12, 32, 2e-4, 0.05, 256


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
    for i in range(0, len(texts), 64):
        out.append(_embed(model, tok, texts[i:i + 64], torch).numpy())
    return np.concatenate(out, 0).astype(np.float16)


def main() -> None:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        print(f"ERROR: needs [embed] extra: {exc}", file=sys.stderr); raise SystemExit(2)
    torch.manual_seed(SEED); np.random.seed(SEED)
    EMB.mkdir(parents=True, exist_ok=True)

    ds, meta = load_scale_crossrepo()
    by = ds.by_id()
    ids = [a.id for a in ds.artifacts]
    texts = [a.text or a.id for a in ds.artifacts]
    pairs = [(by[iss].text or iss, by[pr].text or pr) for pr, iss in ds.fixes
             if iss in by and pr in by and by[iss].split == "train"]
    print(f"scale: {len(ids)} records, {len(pairs)} train pairs, "
          f"{len(meta['train_repos'])} train / {len(meta['test_repos'])} test repos", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(MODEL)
    # Frozen cache.
    base = AutoModel.from_pretrained(MODEL); base.eval()
    np.savez_compressed(EMB / "minilm-l6-v2.npz", ids=np.array(ids),
                        vectors=_embed_all(base, tok, texts, torch), model=MODEL)
    print("wrote frozen cache", file=sys.stderr)

    # LoRA fine-tune.
    model = get_peft_model(AutoModel.from_pretrained(MODEL), LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["query", "key", "value"]))
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    rng = np.random.default_rng(SEED)
    for ep in range(EPOCHS):
        order = rng.permutation(len(pairs))
        tot = 0.0
        for i in range(0, len(pairs), BATCH):
            idx = order[i:i + BATCH]
            if len(idx) < 2:
                continue
            a = _embed(model, tok, [pairs[j][0] for j in idx], torch, grad=True)
            b = _embed(model, tok, [pairs[j][1] for j in idx], torch, grad=True)
            sims = (a @ b.T) / TEMP
            lab = torch.arange(a.shape[0])
            loss = (torch.nn.functional.cross_entropy(sims, lab)
                    + torch.nn.functional.cross_entropy(sims.T, lab)) / 2
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        print(f"  epoch {ep + 1}/{EPOCHS} loss={tot:.3f}", file=sys.stderr)
    model.eval()
    np.savez_compressed(EMB / "minilm-lora.npz", ids=np.array(ids),
                        vectors=_embed_all(model, tok, texts, torch), model=MODEL + "+lora")
    print("wrote LoRA cache -> data/scale/embeddings/")


if __name__ == "__main__":
    main()
