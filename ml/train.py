"""L3 -> L4 -> L5 curriculum training loop (Section 5), warm-starting
each stage from the previous via net2net widening (see lstm_model.py)."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.dataset import OpponentActionDataset, collate_batch, load_corpus_rows
from ml.lstm_model import build_model, widen_warm_start


def train_one_stage(level, corpus_dir, warm_start_from=None, epochs=3, batch_size=64, lr=1e-3, device="cpu"):
    train_rows = load_corpus_rows(corpus_dir, "train", level)
    val_rows = load_corpus_rows(corpus_dir, "val", level)
    if train_rows.empty:
        raise ValueError(f"No training rows found for ML level {level} -- check corpus/stage filter.")

    train_ds = OpponentActionDataset(train_rows)
    val_ds = OpponentActionDataset(val_rows) if not val_rows.empty else None

    model = build_model(level)
    if warm_start_from is not None:
        widen_warm_start(model, warm_start_from)
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_batch) if val_ds else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    curve = {"train": [], "val": []}

    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for x, lengths, y in train_loader:
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            n += len(y)
        train_loss = total / max(n, 1)
        curve["train"].append(train_loss)

        val_loss = None
        if val_loader is not None:
            model.eval()
            total_v, n_v = 0.0, 0
            with torch.no_grad():
                for x, lengths, y in val_loader:
                    loss = loss_fn(model(x, lengths), y)
                    total_v += loss.item() * len(y)
                    n_v += len(y)
            val_loss = total_v / max(n_v, 1)
        curve["val"].append(val_loss)
        print(f"[L{level}] epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss}")

    return model, curve


def run_curriculum(corpus_dir: str, epochs_per_stage: int = 3) -> dict:
    model_l3, curve_l3 = train_one_stage(3, corpus_dir, warm_start_from=None, epochs=epochs_per_stage)
    model_l4, curve_l4 = train_one_stage(4, corpus_dir, warm_start_from=model_l3, epochs=epochs_per_stage)
    model_l5, curve_l5 = train_one_stage(5, corpus_dir, warm_start_from=model_l4, epochs=epochs_per_stage)
    return {3: (model_l3, curve_l3), 4: (model_l4, curve_l4), 5: (model_l5, curve_l5)}