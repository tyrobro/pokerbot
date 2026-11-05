"""Loads corpus shards filtered by curriculum stage (Section 5) and
train/val split (Sprint 11), builds (sequence, label) training examples."""

import glob
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from ml.features import build_opponent_sequences

# Section 5: L3 trains on hands involving math levels 1-2 (weak
# opponents) only; L4 on levels 3-4; L5 on the full corpus.
STAGE_LEVEL_FILTERS = {3: {"l1", "l2"}, 4: {"l3", "l4"}, 5: None}


def _load_split_unit_ids(output_dir: str, split: str) -> set:
    path = Path(output_dir) / "train_val_split.txt"
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s, uid = line.strip().split("\t")
            if s == split:
                ids.add(uid)
    return ids


def load_corpus_rows(output_dir: str, split: str, stage_level: int) -> pd.DataFrame:
    unit_ids = _load_split_unit_ids(output_dir, split)
    allowed = STAGE_LEVEL_FILTERS[stage_level]
    frames = []
    for path in sorted(glob.glob(str(Path(output_dir) / "shards" / "*.parquet"))):
        df = pd.read_parquet(path)
        if df.empty or df.iloc[0]["match_id"] not in unit_ids:
            continue
        if allowed is None or (df.iloc[0]["agent_a_level"] in allowed and df.iloc[0]["agent_b_level"] in allowed):
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class OpponentActionDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, max_seq_len: int = 20):
        self.examples = []
        for _, row in rows.iterrows():
            for player in ("A", "B"):
                for seq, label in build_opponent_sequences(row, player):
                    self.examples.append((seq[-max_seq_len:], label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        seq, label = self.examples[idx]
        return torch.tensor(seq, dtype=torch.float32), label


def collate_batch(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    max_len = max(lengths).item()
    feature_dim = seqs[0].shape[1]
    padded = torch.zeros(len(seqs), max_len, feature_dim)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s
    return padded, lengths, torch.tensor(labels, dtype=torch.long)