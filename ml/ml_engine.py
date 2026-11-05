"""ML engine levels 1-5 (Section 4.1): one interface,
read_opponent(opponent_actions) -> (exploit_signal, confidence), so
GridAgent doesn't need per-level branching."""

import numpy as np
import onnxruntime as ort

from ml.exploit_signal import ml_distribution_to_exploit_signal
from ml.features import encode_action
from ml.rule_based import MLLevel1, MLLevel2


class MLLevelLSTM:
    """ML levels 3-5: wraps an exported ONNX model (ml/onnx/level{k}.onnx,
    Sprint 12's curriculum output)."""

    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path)

    def read_opponent(self, opponent_actions: list) -> tuple:
        if not opponent_actions:
            return 0.0, 0.0  # no history yet -> no signal, no confidence
        pot_before = opponent_actions[0].get("pot_before", 1.0)
        vecs = [encode_action(a, pot_before) for a in opponent_actions]
        x = np.array([vecs], dtype=np.float32)  # (1, seq_len, FEATURE_DIM)
        logits = self.session.run(None, {"opponent_action_sequence": x})[0][0]
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        return ml_distribution_to_exploit_signal(probs.tolist())


def build_ml_engine(level: int, onnx_dir: str = "ml/onnx"):
    if level == 1:
        return MLLevel1()
    if level == 2:
        return MLLevel2()
    if level in (3, 4, 5):
        return MLLevelLSTM(f"{onnx_dir}/level{level}.onnx")
    raise ValueError(f"unknown ML level {level}")