"""demos/demo5_jevball_tactics/tactical_scenarios.py

Curated Tactical Scenarios for JevBall Decision Arena and Benchmark Evaluation.
Each scenario sets up physical positions, tactical dilemmas, and ground-truth tactical benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
from .pitch_sim import Ball, MatchState, Player


@dataclass
class TacticalScenario:
    """Defines a match tactical situation and expected decision traits."""
    id: str
    name: str
    category: str
    description: str
    dilemma_description: str
    expected_optimal_type: str  # 'through', 'shoot', 'pass_safe', 'switch'
    initial_state_builder: Any


def _build_fast_counter_scenario() -> MatchState:
    """Midfield Transition: H10 Playmaker has ball at center circle, H9 Striker darting behind CBs."""
    state = MatchState(
        minute=67,
        second=42.0,
        score_home=1,
        score_away=1,
        phase="COUNTER_ATTACK",
        ball=Ball(x=64.0, y=34.0, carrier_id="H10"),
        active_carrier_id="H10",
        last_event="⚡ Turnover! H10 drives forward into counter-attack space",
    )
    players = [
        # Home
        Player("H1", "home", 1, "Alisson", "GK", 10.0, 34.0),
        Player("H4", "home", 4, "Van Dijk", "CB", 38.0, 30.0),
        Player("H6", "home", 6, "Rodri", "CM", 48.0, 26.0),
        Player("H7", "home", 7, "Saka", "RW", 72.0, 10.0),
        Player("H8", "home", 8, "De Bruyne", "CM", 56.0, 48.0),
        Player("H9", "home", 9, "Haaland", "ST", 84.0, 32.0),
        Player("H10", "home", 10, "Messi", "CAM", 64.0, 34.0),
        Player("H11", "home", 11, "Vinicius", "LW", 75.0, 58.0),
        # Away
        Player("A1", "away", 1, "Courtois", "GK", 100.0, 34.0),
        Player("A2", "away", 2, "Carvajal", "RB", 76.0, 56.0),
        Player("A3", "away", 3, "Davies", "LB", 74.0, 14.0),
        Player("A4", "away", 4, "Rudiger", "CB", 85.5, 29.0),
        Player("A5", "away", 5, "Militao", "CB", 86.0, 39.0),
        Player("A6", "away", 6, "Rice", "CM", 60.5, 34.0),
    ]
    for p in players:
        state.players[p.id] = p
    return state


def _build_box_finish_scenario() -> MatchState:
    """Penalty Box Finishing: H9 Striker receives inside 18-yard box, GK cutting angle, H8 cutback arriving."""
    state = MatchState(
        minute=88,
        second=15.0,
        score_home=2,
        score_away=2,
        phase="BOX_FINISHING",
        ball=Ball(x=92.0, y=28.0, carrier_id="H9"),
        active_carrier_id="H9",
        last_event="H7 crossed low into penalty box, H9 takes control",
    )
    players = [
        # Home
        Player("H1", "home", 1, "Alisson", "GK", 10.0, 34.0),
        Player("H7", "home", 7, "Saka", "RW", 95.0, 12.0),
        Player("H8", "home", 8, "De Bruyne", "CM", 86.0, 35.0),  # Arriving on edge of box for cutback
        Player("H9", "home", 9, "Haaland", "ST", 92.0, 28.0),
        Player("H10", "home", 10, "Messi", "CAM", 85.0, 24.0),
        Player("H11", "home", 11, "Vinicius", "LW", 93.0, 50.0),
        # Away
        Player("A1", "away", 1, "Courtois", "GK", 102.5, 31.5),  # GK committed to near post
        Player("A4", "away", 4, "Rudiger", "CB", 94.0, 33.0),
        Player("A5", "away", 5, "Militao", "CB", 90.0, 24.0),
    ]
    for p in players:
        state.players[p.id] = p
    return state


def _build_high_press_scenario() -> MatchState:
    """Defensive Third Escape: H3 Left-back trapped near corner flag by 2 pressing wingers."""
    state = MatchState(
        minute=23,
        second=10.0,
        score_home=0,
        score_away=0,
        phase="DEFENSIVE_ESCAPE",
        ball=Ball(x=22.0, y=60.0, carrier_id="H3"),
        active_carrier_id="H3",
        last_event="High press trap triggered: A7 & A10 closing down H3",
    )
    players = [
        # Home
        Player("H1", "home", 1, "Alisson", "GK", 8.0, 34.0),  # Open backpass
        Player("H3", "home", 3, "Robertson", "LB", 22.0, 60.0),
        Player("H4", "home", 4, "Van Dijk", "CB", 24.0, 38.0), # Safe central pass
        Player("H6", "home", 6, "Rodri", "CM", 36.0, 48.0),   # Forward passing lane
        # Away
        Player("A7", "away", 7, "Mbappe", "RW", 24.5, 58.5),  # Immediate pressing forward
        Player("A10", "away", 10, "Modric", "CAM", 28.0, 52.0), # Blocking forward pass to H6
    ]
    for p in players:
        state.players[p.id] = p
    return state


def _build_deep_block_switch_scenario() -> MatchState:
    """Deep Low Block: Center is congested with 8 defenders, far-side winger H11 wide open."""
    state = MatchState(
        minute=54,
        second=30.0,
        score_home=0,
        score_away=0,
        phase="DEEP_BLOCK_BREAKING",
        ball=Ball(x=72.0, y=16.0, carrier_id="H7"),
        active_carrier_id="H7",
        last_event="Central corridor congested, H7 looks across pitch",
    )
    players = [
        # Home
        Player("H6", "home", 6, "Rodri", "CM", 60.0, 28.0),
        Player("H7", "home", 7, "Saka", "RW", 72.0, 16.0),
        Player("H8", "home", 8, "De Bruyne", "CM", 70.0, 30.0),
        Player("H9", "home", 9, "Haaland", "ST", 88.0, 33.0),
        Player("H11", "home", 11, "Vinicius", "LW", 82.0, 62.0), # WIDE OPEN ON WEAK SIDE!
        # Away (Packed box)
        Player("A1", "away", 1, "Courtois", "GK", 102.0, 34.0),
        Player("A2", "away", 2, "Carvajal", "RB", 85.0, 50.0),
        Player("A3", "away", 3, "Davies", "LB", 75.0, 16.0),
        Player("A4", "away", 4, "Rudiger", "CB", 87.0, 31.0),
        Player("A5", "away", 5, "Militao", "CB", 88.0, 36.0),
        Player("A6", "away", 6, "Rice", "CM", 78.0, 26.0),
        Player("A8", "away", 8, "Bellingham", "CM", 80.0, 34.0),
    ]
    for p in players:
        state.players[p.id] = p
    return state


SCENARIOS: Dict[str, TacticalScenario] = {
    "fast_counter": TacticalScenario(
        id="fast_counter",
        name="Fast Counter Channel Through Ball",
        category="ATTACKING_TRANSITION",
        description="Midfield turnover transition to attack: dual center-backs push high to set offside trap, striker makes diagonal run into channel space.",
        dilemma_description="Dilemma: Risk through-ball to striker for 1v1 vs safe square pass to reset shape vs solo dribble carry.",
        expected_optimal_type="through",
        initial_state_builder=_build_fast_counter_scenario,
    ),
    "box_finish": TacticalScenario(
        id="box_finish",
        name="Penalty Box Clinical Finish & Cutback Decision",
        category="FINAL_THIRD_EXECUTION",
        description="Penetrating right byline inside box: goalkeeper covers near post, far post semi-open, trailing midfielder arriving unmarked at spot.",
        dilemma_description="Dilemma: Narrow-angle far-post drive vs low cutback cross for unmarked tap-in vs shielding for contact foul.",
        expected_optimal_type="shoot",
        initial_state_builder=_build_box_finish_scenario,
    ),
    "high_press_escape": TacticalScenario(
        id="high_press_escape",
        name="High Press Trap De-escalation & Buildup",
        category="DEFENSIVE_BUILDUP",
        description="Trapped in defensive corner by two-man closing press: forward passing channels locked, interception window only 0.4s.",
        dilemma_description="Dilemma: Back-pass to goalkeeper to clear safety vs speculative long clearance vs risky turn dribble.",
        expected_optimal_type="pass_safe",
        initial_state_builder=_build_high_press_scenario,
    ),
    "deep_block_switch": TacticalScenario(
        id="deep_block_switch",
        name="Deep Low Block Switch to Weak Side Flank",
        category="CHANCE_CREATION",
        description="Opponent parks 8-man low block in central box: right-flank progression congested, left-flank winger wide open in isolated space.",
        dilemma_description="Dilemma: Long diagonal cross-field switch to isolated winger vs congested central penetration vs speculative long shot.",
        expected_optimal_type="pass",
        initial_state_builder=_build_deep_block_switch_scenario,
    ),
}
