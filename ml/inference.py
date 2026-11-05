"""Inference latency profiling (Section 7's compute note: ML inference
cost is the most likely bottleneck -- profile before committing to the
tournament run's expected runtime)."""

import time

import numpy as np
import onnxruntime as ort

from ml.lstm_model import FEATURE_DIM


def profile_inference_latency(onnx_path: str, seq_len: int = 10, n_trials: int = 200) -> dict:
    session = ort.InferenceSession(onnx_path)
    x = np.random.randn(1, seq_len, FEATURE_DIM).astype(np.float32)

    for _ in range(10):  # warmup
        session.run(None, {"opponent_action_sequence": x})

    times_ms = []
    for _ in range(n_trials):
        start = time.perf_counter()
        session.run(None, {"opponent_action_sequence": x})
        times_ms.append((time.perf_counter() - start) * 1000)

    times_ms.sort()
    return {
        "mean_ms": sum(times_ms) / len(times_ms),
        "p50_ms": times_ms[len(times_ms) // 2],
        "p99_ms": times_ms[int(len(times_ms) * 0.99)],
        "n_trials": n_trials,
    }