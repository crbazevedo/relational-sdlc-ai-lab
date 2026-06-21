#!/usr/bin/env python3
"""LoRA fine-tune a small embedder with the relation (contrastive) loss.

This is the first experiment where the relational contribution lives INSIDE the
representation, not as a bolt-on head (R8 showed a frozen-embedder head can't help).
We LoRA-tune MiniLM on TRAIN-repo `fixes` pairs with an InfoNCE / multiple-negatives
loss (in-batch negatives), then cache the tuned embeddings. Evaluation
(run_finetune_ablation.py) is numpy-only on the committed cache, so the win/loss vs
the frozen embedder is reproducible without torch.

Discipline (.relsdlc/playbooks/finetuning-discipline.md): train on train repos only;
LoRA not full FT; seed everything; cache + commit; honest delta vs the frozen control.

Run:  python data/pilot/finetune_embed.py     # needs: pip install -e '.[embed]'
Out:  data/pilot/embeddings/minilm-lora.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from run_crossrepo_ablation import load_pilot_crossrepo  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUT = HERE / "embeddings" / "minilm-lora.npz"
SEED = 0
EPOCHS = 12
BATCH = 32
LR = 2e-4
MAX_LEN = 256
TEMP = 0.05


def _mean_pool(last_hidden, mask):
    import torch
    m = mask.unsqueeze(-1).float()
    return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def _embed(model, tok, texts, torch):
    enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    out = model(**enc)
    emb = _mean_pool(out.last_hidden_state, enc["attention_mask"])
    return torch.nn.functional.normalize(emb, p=2, dim=1)


def main() -> None:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        print(f"ERROR: needs the embed extra (pip install -e '.[embed]'): {exc}", file=sys.stderr)
        raise SystemExit(2)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ds, meta = load_pilot_crossrepo()
    by_id = ds.by_id()
    pairs = [(by_id[iss].text or iss, by_id[pr].text or pr)
             for pr, iss in ds.fixes
             if iss in by_id and pr in by_id and by_id[iss].split == "train"]
    print(f"train pairs (train repos only): {len(pairs)}", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModel.from_pretrained(MODEL)
    lora = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=8, lora_alpha=16,
                      lora_dropout=0.05, target_modules=["query", "key", "value"])
    model = get_peft_model(base, lora)
    model.train()
    model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    rng = np.random.default_rng(SEED)
    n = len(pairs)
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        total = 0.0
        for i in range(0, n, BATCH):
            idx = order[i:i + BATCH]
            if len(idx) < 2:
                continue
            q = [pairs[j][0] for j in idx]
            d = [pairs[j][1] for j in idx]
            a = _embed(model, tok, q, torch)
            b = _embed(model, tok, d, torch)
            sims = (a @ b.T) / TEMP
            labels = torch.arange(a.shape[0])
            loss = (torch.nn.functional.cross_entropy(sims, labels)
                    + torch.nn.functional.cross_entropy(sims.T, labels)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        print(f"  epoch {epoch + 1}/{EPOCHS} loss={total:.4f}", file=sys.stderr)

    model.eval()
    ids = [a.id for a in ds.artifacts]
    texts = [a.text or a.id for a in ds.artifacts]
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            vecs.append(_embed(model, tok, texts[i:i + 64], torch).numpy())
    matrix = np.concatenate(vecs, axis=0).astype(np.float16)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, ids=np.array(ids), vectors=matrix,
                        model=MODEL + "+lora", base=MODEL)
    print(f"wrote {matrix.shape} LoRA-tuned embeddings -> {OUT}")


if __name__ == "__main__":
    main()
