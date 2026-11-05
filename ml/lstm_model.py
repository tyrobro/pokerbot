"""Single LSTM architecture family (Section 4.1): levels 3-5 share this
exact architecture, differing only in capacity (hidden_size) and
training curriculum stage -- never in architecture family, per Section
4.1's explicit requirement.
"""

import torch
import torch.nn as nn

ACTION_TYPES = ["FOLD", "CHECK", "CALL", "RAISE", "ALL_IN"]
STREETS = ["PREFLOP", "FLOP", "TURN", "RIVER"]
FEATURE_DIM = len(STREETS) + len(ACTION_TYPES) + 1  # street one-hot + action one-hot + normalized size


class OpponentLSTM(nn.Module):
    """Predicts an opponent's NEXT action from their action sequence so
    far this hand -- self-supervised, no extra labels needed beyond the
    corpus. The output distribution's confidence/shape is Section 4.1's
    exploit_signal/confidence pair once Sprint 13 wires this in."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=FEATURE_DIM, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, len(ACTION_TYPES))

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.head(h_n[-1])


LEVEL_HIDDEN_SIZE = {3: 16, 4: 32, 5: 64}


def build_model(level: int) -> OpponentLSTM:
    return OpponentLSTM(hidden_size=LEVEL_HIDDEN_SIZE[level])


def _net2net_mapping(old_size: int, new_size: int, generator: torch.Generator) -> torch.Tensor:
    """New unit i<old_size maps to itself (identity); new_size-old_size
    additional units are RANDOM DUPLICATES of existing old units (not
    fresh random units) -- this is what makes the widening function-
    preserving at init, per Chen et al.'s Net2WiderNet: every new unit
    starts out computing the same thing some existing unit already
    computes, rather than injecting noise."""
    assert new_size >= old_size
    identity = torch.arange(old_size)
    extra = torch.randint(0, old_size, (new_size - old_size,), generator=generator)
    return torch.cat([identity, extra])


def _duplication_counts(mapping: torch.Tensor, old_size: int) -> torch.Tensor:
    """How many times each original unit (0..old_size-1) appears in the
    mapping -- needed to rescale INCOMING weights from duplicated units
    so their combined contribution still equals the original single
    unit's contribution (the other half of function-preservation)."""
    counts = torch.zeros(old_size)
    for idx in mapping:
        counts[idx] += 1
    return counts


def _widen_output_rows(weight: torch.Tensor, mapping: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Widens along dim 0 (output units) by duplication. Rows are NOT
    rescaled here -- rescaling belongs on the INCOMING side of a
    duplicated unit (see _widen_input_cols_rescaled), not the outgoing
    gate-computation side. Small noise is added to duplicate rows only,
    for symmetry-breaking (identical duplicate units would otherwise
    receive identical gradients forever and never differentiate)."""
    new_weight = weight[mapping].clone()
    old_size = weight.shape[0]
    is_duplicate = torch.arange(len(mapping)) >= old_size
    new_weight[is_duplicate] += torch.randn_like(new_weight[is_duplicate]) * noise_std
    return new_weight


def _widen_input_cols_rescaled(weight: torch.Tensor, mapping: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Widens along dim 1 (input/recurrent-feed columns) by duplication,
    then divides each column by how many times its source unit was
    duplicated overall -- this is the actual function-preserving step:
    if old unit k is duplicated into 3 new units, each of those 3 carries
    1/3 of the original incoming weight, so summing their contributions
    reconstructs exactly what the single original unit contributed."""
    new_weight = weight[:, mapping].clone()
    scale = counts[mapping]  # one scale factor per new column
    new_weight = new_weight / scale.unsqueeze(0)
    return new_weight


def widen_warm_start(new_model: OpponentLSTM, old_model: OpponentLSTM, seed: int = 0) -> None:
    """Net2Net widening (Chen, Goodfellow & Shlens 2016), applied per-gate
    to the LSTM's stacked gate structure: new hidden units are random
    DUPLICATES of existing ones (not fresh random init), with outgoing
    weights left as near-identical copies (+ small symmetry-breaking
    noise) and incoming weights rescaled by the duplication count. This
    makes the widened network compute approximately the same function as
    the smaller one immediately after widening, before any further
    training -- which is the actual point of warm-starting (preserve
    what the previous curriculum stage learned), not just "copy what
    fits and randomize the rest."

    Known approximation (worth stating precisely): the LSTM's hidden
    dimension appears on BOTH sides of the recurrent weight (weight_hh:
    output rows via the 4 stacked gates, AND input columns via the
    recurrent feedback), which the base Net2WiderNet recipe (derived for
    plain feedforward layers) doesn't directly address. Handled here by
    applying row-duplication (gate outputs) and column-duplication+
    rescaling (recurrent input) with the SAME mapping, so a given new
    hidden unit is consistently "the same unit" on both sides. This is a
    reasonable, standard extension, not a literature-verified RNN-exact
    guarantee -- flagged as an approximation, not claimed as exact.
    """
    old_hidden = old_model.hidden_size
    new_hidden = new_model.hidden_size
    generator = torch.Generator().manual_seed(seed)
    mapping = _net2net_mapping(old_hidden, new_hidden, generator)
    counts = _duplication_counts(mapping, old_hidden)

    with torch.no_grad():
        old_params = dict(old_model.lstm.named_parameters())
        for name, new_param in new_model.lstm.named_parameters():
            old_param = old_params[name]

            if "weight_ih" in name:
                # Rows = 4 stacked gates x hidden; input dim (FEATURE_DIM)
                # is unchanged across levels, so only widen the gate/
                # hidden dimension (rows), duplicated per gate block.
                gate_blocks = [
                    _widen_output_rows(old_param[g * old_hidden : (g + 1) * old_hidden], mapping, noise_std=0.01)
                    for g in range(4)
                ]
                new_param.copy_(torch.cat(gate_blocks, dim=0))

            elif "weight_hh" in name:
                # BOTH sides widen: rows (gate outputs, duplicated+noise)
                # AND columns (recurrent input, duplicated+rescaled).
                gate_blocks = []
                for g in range(4):
                    old_gate = old_param[g * old_hidden : (g + 1) * old_hidden]
                    rows_widened = _widen_output_rows(old_gate, mapping, noise_std=0.01)
                    both_widened = _widen_input_cols_rescaled(rows_widened, mapping, counts)
                    gate_blocks.append(both_widened)
                new_param.copy_(torch.cat(gate_blocks, dim=0))

            elif "bias" in name:
                gate_blocks = [
                    _widen_output_rows(old_param[g * old_hidden : (g + 1) * old_hidden], mapping, noise_std=0.01)
                    for g in range(4)
                ]
                new_param.copy_(torch.cat(gate_blocks, dim=0))

        # head.weight: columns = hidden units feeding the final linear
        # layer -> duplicate + rescale (function-preserving). head.bias
        # is untouched (its size is the fixed action count, independent
        # of hidden size).
        new_model.head.weight.copy_(_widen_input_cols_rescaled(old_model.head.weight, mapping, counts))
        new_model.head.bias.copy_(old_model.head.bias)