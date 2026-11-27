"""demos/demo5_jevball_tactics

Tractus JevBall Real-Time Multi-Agent Tactical Decision Simulator.
Spatial multi-agent decision making on 105m x 68m football pitch with up to 14 typed candidate actions,
deterministic 2D kinematics, and real Metal GPU forward passes answering JEV System One contracts.
"""

from .pitch_sim import Ball, MatchState, PitchSimulator, Player
from .candidate_generator import TacticalCandidate, TacticalCandidateGenerator
from .prompt_contract import JevBallDecisionEngine, JevBallDecisionResult
from .tactical_scenarios import SCENARIOS, TacticalScenario
from .radar_display import TacticalRadarRenderer

__all__ = [
    "Ball",
    "Player",
    "MatchState",
    "PitchSimulator",
    "TacticalCandidate",
    "TacticalCandidateGenerator",
    "JevBallDecisionEngine",
    "JevBallDecisionResult",
    "SCENARIOS",
    "TacticalScenario",
    "TacticalRadarRenderer",
]
