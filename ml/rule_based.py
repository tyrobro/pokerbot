"""ML engine levels 1-2 (Section 5, steps 2-3): "no training -- rule-
based, implemented directly" -- not learned models."""


class MLLevel1:
    """No opponent modeling -- ignores all history."""

    def read_opponent(self, opponent_actions: list) -> tuple:
        return 0.0, 0.0  # exploit_signal, confidence


class MLLevel2:
    """Simple running statistics (fold-to-bet %, aggression factor) --
    counting, not a trained model. Each action dict needs action_type
    and, for fold-to-bet%, a faced_bet bool."""

    def read_opponent(self, opponent_actions: list) -> tuple:
        if not opponent_actions:
            return 0.0, 0.0
        aggressive = sum(1 for a in opponent_actions if a["action_type"] in ("RAISE", "ALL_IN"))
        folds = sum(1 for a in opponent_actions if a["action_type"] == "FOLD")
        faced_bets = sum(1 for a in opponent_actions if a.get("faced_bet"))
        aggression_factor = aggressive / len(opponent_actions)
        fold_to_bet_pct = folds / faced_bets if faced_bets > 0 else 0.0
        exploit_signal = aggression_factor - fold_to_bet_pct
        confidence = min(len(opponent_actions) / 10.0, 1.0)
        return exploit_signal, confidence