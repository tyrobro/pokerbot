import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.math_agents import MathLevel3Agent, MathLevel4Agent
from orchestrator.round_robin import play_pairing
from orchestrator.calibration import DEFAULT_CONFIG

config = dict(DEFAULT_CONFIG)
agent_a = MathLevel3Agent(seed=1)
agent_b = MathLevel4Agent(seed=2)
result = play_pairing(agent_a, agent_b, "math_l3", "math_l4", config, logs_dir="logs/debug_tie")
print(result["pairing_score_a"], result["pairing_score_b"])
for leg in result["leg_results"]:
    print("hands_played:", leg["hands_played"], "final:", leg["final_stack_a"], leg["final_stack_b"])