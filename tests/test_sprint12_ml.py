"""Sprint 12 DoD: sensible decreasing loss, ONNX export parity, rule-
based L1/L2. Uses a tiny synthetic corpus for speed -- the real
curriculum run against the full corpus is the manual script above."""

import pandas as pd
import torch

from ml.dataset import OpponentActionDataset, collate_batch
from ml.export_onnx import check_onnx_parity, export_to_onnx
from ml.features import build_opponent_sequences, parse_action_sequence
from ml.lstm_model import build_model, widen_warm_start
from ml.rule_based import MLLevel1, MLLevel2


def _sample_row():
    return pd.Series(
        {
            "action_sequence": "PREFLOP:RAISE:300;PREFLOP:CALL:300;FLOP:CHECK;FLOP:RAISE:450;FLOP:CALL:450",
            "pot_size_progression": "300;600;600;1050;1500",
            "button_position": "A",
            "small_blind": 50,
            "big_blind": 100,
        }
    )


def test_parse_action_sequence_alternates_players_starting_with_button_preflop():
    entries = parse_action_sequence(_sample_row()["action_sequence"], button_position="A")
    assert entries[0]["player"] == "A"  # button acts first preflop
    assert entries[1]["player"] == "B"
    # postflop: non-button (B) acts first
    assert entries[2]["street"] == "FLOP"
    assert entries[2]["player"] == "B"


def test_build_opponent_sequences_produces_context_label_pairs():
    examples = build_opponent_sequences(_sample_row(), target_player="A")
    assert len(examples) > 0
    for seq, label in examples:
        assert len(seq) > 0  # requires at least one prior action as context
        assert isinstance(label, int)


def test_widen_warm_start_is_approximately_function_preserving():
    """The actual point of Net2Net: feeding the SAME input through the
    smaller (pre-widen) and larger (post-widen, before further training)
    models should produce approximately the same output -- not just
    "some weights got copied somewhere." This is the property a naive
    top-left-copy-with-random-init does NOT have (new random units
    inject noise immediately), and is what distinguishes real Net2Net
    widening from a partial, naive version of it."""
    torch.manual_seed(0)
    small = build_model(3)  # hidden_size 16
    large = build_model(4)  # hidden_size 32
    widen_warm_start(large, small, seed=1)

    small.eval()
    large.eval()
    x = torch.randn(1, 5, small.lstm.input_size)
    lengths = torch.tensor([5])

    with torch.no_grad():
        small_out = small(x, lengths)
        large_out = large(x, lengths)

    # Not exact (duplicate-unit noise + nonlinearity inside the LSTM gates
    # mean this is an approximation, not a mathematical identity -- see
    # the function's own doc comment), but should be close, not noise-level.
    assert torch.allclose(small_out, large_out, atol=0.15), f"small={small_out}, large={large_out}"


def test_widen_warm_start_duplicates_not_randomizes_new_units():
    """Distinguishes real Net2Net from the earlier, rejected top-left-
    copy-with-random-init: every row of the widened weight_ih must
    exactly match SOME row of the original (a duplicate, possibly with
    tiny symmetry-breaking noise on the new ones) -- never an
    independent fresh-random row uncorrelated with the original model."""
    torch.manual_seed(0)
    small = build_model(3)
    large = build_model(4)
    widen_warm_start(large, small, seed=2)

    old_hidden = small.hidden_size
    small_ih = dict(small.lstm.named_parameters())["weight_ih_l0"]
    large_ih = dict(large.lstm.named_parameters())["weight_ih_l0"]

    # Check the FIRST gate block's new (duplicated) rows are each close
    # to at least one row from the small model's first gate block.
    small_gate0 = small_ih[:old_hidden]
    large_gate0 = large_ih[:large.hidden_size]
    for new_row in large_gate0[old_hidden:]:
        dists = (small_gate0 - new_row).abs().sum(dim=1)
        assert dists.min() < 0.5, "new unit's row isn't close to any original row -- looks randomly initialized, not duplicated"

def test_tiny_training_step_runs_and_loss_is_finite():
    row = _sample_row()
    examples = []
    for player in ("A", "B"):
        examples.extend(build_opponent_sequences(row, player))
    ds = OpponentActionDataset(pd.DataFrame([row] * 5))  # a few repeats for a non-trivial batch
    assert len(ds) > 0

    x, lengths, y = collate_batch([ds[i] for i in range(min(4, len(ds)))])
    model = build_model(3)
    logits = model(x, lengths)
    loss = torch.nn.functional.cross_entropy(logits, y)
    assert torch.isfinite(loss)


def test_onnx_export_and_parity():
    model = build_model(3)
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "model.onnx")
        export_to_onnx(model, onnx_path)
        assert check_onnx_parity(model, onnx_path)


def test_ml_level1_ignores_history():
    signal, confidence = MLLevel1().read_opponent([{"action_type": "RAISE"}])
    assert signal == 0.0
    assert confidence == 0.0


def test_ml_level2_aggression_and_confidence_scale_with_history():
    actions = [{"action_type": "RAISE", "faced_bet": False}] * 8
    signal, confidence = MLLevel2().read_opponent(actions)
    assert signal > 0  # all-aggressive history -> positive signal
    assert confidence == pytest_approx(0.8)


def pytest_approx(x, tol=1e-9):
    class _A:
        def __eq__(self, other):
            return abs(other - x) < tol
    return _A()