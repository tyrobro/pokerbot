"""Sprint 9 DoD: fixed-seed calibration round-robin, 10 agents (5 math
levels + 5 PD archetypes) = 45 pairings x 2 legs = 90 match legs."""

from agents.math_agents import MathLevel1Agent, MathLevel2Agent, MathLevel3Agent, MathLevel4Agent, MathLevel5Agent
from agents.pd_archetypes import AlwaysCooperateAgent, AlwaysDefectAgent, GrudgerAgent, PavlovAgent, TitForTatAgent
from orchestrator.round_robin import run_round_robin

CALIBRATION_AGENT_FACTORIES = {
    "math_l1": lambda seed: MathLevel1Agent(),
    "math_l2": lambda seed: MathLevel2Agent(),
    "math_l3": lambda seed: MathLevel3Agent(seed=seed),
    "math_l4": lambda seed: MathLevel4Agent(seed=seed),
    "math_l5": lambda seed: MathLevel5Agent(seed=seed, train_iterations=30),
    "pd_tit_for_tat": lambda seed: TitForTatAgent(),
    "pd_always_defect": lambda seed: AlwaysDefectAgent(),
    "pd_always_cooperate": lambda seed: AlwaysCooperateAgent(seed=seed, train_iterations=30),
    "pd_pavlov": lambda seed: PavlovAgent(),
    "pd_grudger": lambda seed: GrudgerAgent(),
}

# NOTE: max_hands reduced from the production 1000 to 200 here -- the
# Sprint 9 DoD requires completion + reproducibility + a reviewable
# bust-time distribution, not full 1000-hand matches. Revisit at full
# scale alongside CFRConfig::default() once Sprint 11-12 exist.
DEFAULT_CONFIG = {
    "starting_stack": 100_000,
    "initial_small_blind": 50,
    "initial_big_blind": 100,
    "blind_escalation_interval_hands": 20,
    "max_hands": 200,
    "seed": 42,
}


def run_calibration(logs_dir="logs/calibration"):
    return run_round_robin(CALIBRATION_AGENT_FACTORIES, DEFAULT_CONFIG, logs_dir)