"""Sprint 14: the full 30-agent roster -- 25 GridAgent(m,k) combinations
(Section 4.1) + 5 PD archetypes (Section 4.2). Agent factories are
picklable top-level functions (not lambdas) so they work under Windows'
spawn-based multiprocessing (Sprint 11's corpus generation hit this same
constraint).
"""

from agents.grid_agent import GridAgent
from agents.pd_archetypes import (
    AlwaysCooperateAgent,
    AlwaysDefectAgent,
    GrudgerAgent,
    PavlovAgent,
    TitForTatAgent,
)
from ml.blueprint_store import get_shared_l5_blueprint

PD_AGENT_IDS = ["pd_tit_for_tat", "pd_always_defect", "pd_always_cooperate", "pd_pavlov", "pd_grudger"]


def build_agent_id(math_level: int, ml_level: int) -> str:
    return f"grid_m{math_level}_k{ml_level}"


def all_agent_ids() -> list:
    """25 GridAgent combinations + 5 PD archetypes = 30 (Sprint 14's
    DoD roster)."""
    grid_ids = [build_agent_id(m, k) for m in range(1, 6) for k in range(1, 6)]
    return grid_ids + PD_AGENT_IDS


def build_agent(agent_id: str, seed: int):
    """Factory: builds ONE fresh agent instance for agent_id (Section
    4.4: fresh and anonymous per pairing). GridAgent(5,*) and
    AlwaysCooperateAgent share the per-WORKER-PROCESS cached L5
    blueprint (get_shared_l5_blueprint) -- trained at most once per
    process, never once per pairing (Sprint 14 fix)."""
    if agent_id.startswith("grid_m"):
        rest = agent_id[len("grid_m"):]
        m_str, k_str = rest.split("_k")
        math_level, ml_level = int(m_str), int(k_str)
        kwargs = {}
        if math_level == 5:
            kwargs["shared_l5_solver"] = get_shared_l5_blueprint()
        return GridAgent(math_level=math_level, ml_level=ml_level, seed=seed, **kwargs)

    if agent_id == "pd_tit_for_tat":
        return TitForTatAgent()
    if agent_id == "pd_always_defect":
        return AlwaysDefectAgent()
    if agent_id == "pd_always_cooperate":
        return AlwaysCooperateAgent(shared_l5_solver=get_shared_l5_blueprint(), seed=seed)
    if agent_id == "pd_pavlov":
        return PavlovAgent()
    if agent_id == "pd_grudger":
        return GrudgerAgent()

    raise ValueError(f"unknown agent_id {agent_id!r}")