"""Run manually once the corpus exists: python scripts/train_ml_curriculum.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from ml.export_onnx import check_onnx_parity, export_to_onnx
from ml.inference import profile_inference_latency
from ml.train import run_curriculum

if __name__ == "__main__":
    results = run_curriculum("logs/corpus", epochs_per_stage=3)

    Path("ml/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("ml/onnx").mkdir(parents=True, exist_ok=True)

    for level, (model, curve) in results.items():
        torch.save(model.state_dict(), f"ml/checkpoints/level{level}.pt")
        onnx_path = f"ml/onnx/level{level}.onnx"
        export_to_onnx(model, onnx_path)
        parity_ok = check_onnx_parity(model, onnx_path)
        latency = profile_inference_latency(onnx_path)
        print(
            f"L{level}: train_loss={curve['train'][-1]:.4f} val_loss={curve['val'][-1]} "
            f"parity_ok={parity_ok} latency_p50={latency['p50_ms']:.3f}ms"
        )