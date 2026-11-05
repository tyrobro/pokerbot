"""ONNX export + numerical parity check (Sprint 12 DoD)."""

import numpy as np
import onnx
import onnxruntime as ort
import torch

from ml.lstm_model import FEATURE_DIM


class _InferenceWrapper(torch.nn.Module):
    """pack_padded_sequence isn't cleanly ONNX-exportable with dynamic
    lengths; at inference time (batch size 1, real known length) no
    padding is needed, so export a simpler unpacked-LSTM variant."""

    def __init__(self, inner):
        super().__init__()
        self.lstm = inner.lstm
        self.head = inner.head

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])


def export_to_onnx(model, output_path: str, dummy_seq_len: int = 5) -> str:
    model.eval()
    wrapper = _InferenceWrapper(model)
    dummy_x = torch.randn(1, dummy_seq_len, FEATURE_DIM)
    torch.onnx.export(
        wrapper,
        dummy_x,
        output_path,
        input_names=["opponent_action_sequence"],
        output_names=["next_action_logits"],
        dynamic_axes={"opponent_action_sequence": {1: "seq_len"}},
        opset_version=13,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(output_path))
    return output_path


def check_onnx_parity(model, onnx_path: str, seq_len: int = 5, tolerance: float = 1e-4) -> bool:
    model.eval()
    x = torch.randn(1, seq_len, FEATURE_DIM)
    with torch.no_grad():
        _, (h_n, _) = model.lstm(x)
        torch_out = model.head(h_n[-1]).numpy()

    session = ort.InferenceSession(onnx_path)
    onnx_out = session.run(None, {"opponent_action_sequence": x.numpy()})[0]
    return bool(np.allclose(torch_out, onnx_out, atol=tolerance))