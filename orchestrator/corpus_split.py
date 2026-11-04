"""Section 5.8: 80/20 train/val split BY MATCH, not by hand (avoids
leaking one match's dynamics across train/val)."""

import random
from pathlib import Path


def split_corpus_by_match(output_dir: str, val_fraction: float = 0.2, seed: int = 1) -> dict:
    manifest_path = Path(output_dir) / "manifest.txt"
    with open(manifest_path, "r", encoding="utf-8") as f:
        unit_ids = sorted(line.strip() for line in f if line.strip())

    rng = random.Random(seed)
    shuffled = unit_ids[:]
    rng.shuffle(shuffled)

    val_count = int(len(shuffled) * val_fraction)
    val_ids = sorted(shuffled[:val_count])
    train_ids = sorted(shuffled[val_count:])

    split_path = Path(output_dir) / "train_val_split.txt"
    with open(split_path, "w", encoding="utf-8") as f:
        for uid in train_ids:
            f.write(f"train\t{uid}\n")
        for uid in val_ids:
            f.write(f"val\t{uid}\n")

    return {"train_matches": len(train_ids), "val_matches": len(val_ids), "split_path": str(split_path)}