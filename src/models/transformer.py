"""Optional transformer baseline (DistilBERT) evaluated on the SAME val/test split as the classical models.

Writes reports/transformer_metrics.json; src/models/train.py then lists it in the comparison table.
It is a comparison/benchmark only: it is not wired into serving, so it never silently becomes the
production model - promoting it would be an explicit, reviewed step.

Run: python -m src.models.transformer [--train-samples 6000] [--epochs 1]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config, resolve
from src.data.loader import load_dataset, split_dataset
from src.evaluation.metrics import compute_metrics


def predict(model, tok, texts, device, max_len, bs=64):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
            out.append(torch.softmax(model(**enc).logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-samples", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config()
    tc = cfg["training"]["transformer"]
    n, epochs, L = a.train_samples or tc["train_samples"], a.epochs or tc["epochs"], tc["max_length"]
    torch.manual_seed(cfg["project"]["seed"])
    df, _ = load_dataset(cfg)
    train, val, test = split_dataset(df, cfg)
    train = train.sample(min(n, len(train)), random_state=cfg["project"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(tc["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(tc["model_name"], num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    t0 = time.time()
    idx = np.arange(len(train))
    texts, labels = train["text"].tolist(), train["label"].values
    for ep in range(epochs):
        model.train()
        np.random.default_rng(ep).shuffle(idx)
        for s in range(0, len(idx), 16):
            b = idx[s:s + 16]
            enc = tok([texts[i] for i in b], truncation=True, max_length=L, padding=True, return_tensors="pt").to(device)
            loss = model(**enc, labels=torch.tensor(labels[b]).to(device)).loss
            loss.backward(); opt.step(); opt.zero_grad()
            if (s // 16) % 50 == 0:
                print(f"epoch {ep} step {s // 16} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    pv = predict(model, tok, val["text"].tolist(), device, L)
    t1 = time.perf_counter()
    pt = predict(model, tok, test["text"].tolist(), device, L)
    ms = (time.perf_counter() - t1) / len(test) * 1000
    res = {"display_name": f"{tc['model_name']} (fine-tuned on {len(train):,} reviews)", "params": {"epochs": epochs, "max_length": L},
           "val": compute_metrics(val["label"], pv), "test": compute_metrics(test["label"], pt),
           "inference_ms_per_review": ms, "train_seconds": time.time() - t0, "note": "benchmark only; not deployed"}
    (resolve(cfg["paths"]["reports_dir"]) / "transformer_metrics.json").write_text(json.dumps(res, indent=2))
    print("test F1(fake)=%.4f ROC-AUC=%.4f" % (res["test"]["f1_fake"], res["test"]["roc_auc"]))


if __name__ == "__main__":
    main()
