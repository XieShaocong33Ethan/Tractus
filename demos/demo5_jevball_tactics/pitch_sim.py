"""demos/demo5_jevball_tactics/pitch_sim.py

Deterministic 2D Kinematics and Pitch State for JevBall Tactical Simulation.
Models a standard 105m x 68m pitch, 22 independent agents (Home: Blue, Away: Red),
ball physics, possession tracking, and line clearance raycasts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GOAL_Y_MIN = 30.34
GOAL_Y_MAX = 37.66
GOAL_WIDTH = 7.32


@dataclass
class Player:
    """Represents a single football player agent on the pitch."""
    id: str
    team: str  # 'home' or 'away'
    number: int
    name: str
    role: str  # 'GK', 'CB', 'LB', 'RB', 'CM', 'CAM', 'LW', 'RW', 'ST'
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    max_speed: float = 8.5  # m/s
    stamina: float = 100.0

    @property
    def pos(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def dist_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)


@dataclass
class Ball:
    """Represents the football with position and velocity."""
    x: float = 52.5
    y: float = 34.0
    vx: float = 0.0
    vy: float = 0.0
    carrier_id: Optional[str] = None

    @property
    def pos(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def dist_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)


@dataclass
class MatchState:
    """Full snapshot of the match state at a given tick."""
    minute: int = 34
    second: float = 12.0
    score_home: int = 1
    score_away: int = 0
    phase: str = "OPEN_PLAY_ATTACK"  # 'OPEN_PLAY_ATTACK', 'COUNTER_ATTACK', 'HIGH_PRESS', 'BOX_FINISH'
    players: Dict[str, Player] = field(default_factory=dict)
    ball: Ball = field(default_factory=Ball)
    active_carrier_id: Optional[str] = None
    last_event: str = "Match in progress"

    def get_player(self, player_id: str) -> Optional[Player]:
        return self.players.get(player_id)

    @property
    def active_carrier(self) -> Optional[Player]:
        if self.ball.carrier_id and self.ball.carrier_id in self.players:
            return self.players[self.ball.carrier_id]
        if self.active_carrier_id and self.active_carrier_id in self.players:
            return self.players[self.active_carrier_id]
        return None

    def get_team_players(self, team: str) -> List[Player]:
        return [p for p in self.players.values() if p.team == team]

    def offside_line_x(self, defending_team: str) -> float:
        """Computes the offside line (last defender position excluding goalkeeper)."""
        defenders = [p for p in self.players.values() if p.team == defending_team and p.role != "GK"]
        if not defenders:
            return 52.5
        if defending_team == "away":
            # Away defending right goal (x=105m): offside line is max x of second-to-last defender
            xs = sorted([p.x for p in defenders])
            return xs[-1] if xs else 85.0
        else:
            # Home defending left goal (x=0m): offside line is min x
            xs = sorted([p.x for p in defenders])
            return xs[0] if xs else 20.0


class PitchSimulator:
    """Manages pitch physics and initial formation layout."""

    @staticmethod
    def create_default_match() -> MatchState:
        """Creates a realistic match snapshot in the attacking 3rd for Home team."""
        state = MatchState(
            minute=34,
            second=18.4,
            score_home=1,
            score_away=0,
            phase="ATTACKING_TRANSITION",
            ball=Ball(x=68.2, y=34.0, carrier_id="H10"),
            active_carrier_id="H10",
            last_event="H8 won duel, passed forward to H10 Playmaker",
        )

        # Home Team (Blue, attacking left -> right towards x=105)
        home_players = [
            Player("H1", "home", 1, "Alisson", "GK", 6.0, 34.0),
            Player("H2", "home", 2, "Walker", "RB", 42.0, 10.0),
            Player("H3", "home", 3, "Robertson", "LB", 48.0, 58.0),
            Player("H4", "home", 4, "Van Dijk", "CB", 32.0, 26.0),
            Player("H5", "home", 5, "Saliba", "CB", 33.0, 42.0),
            Player("H6", "home", 6, "Rodri", "CM", 52.0, 31.0),
            Player("H7", "home", 7, "Saka", "RW", 74.0, 12.0),
            Player("H8", "home", 8, "De Bruyne", "CM", 58.0, 44.0),
            Player("H9", "home", 9, "Haaland", "ST", 86.5, 33.5),
            Player("H10", "home", 10, "Messi", "CAM", 68.2, 34.0),
            Player("H11", "home", 11, "Vinicius", "LW", 78.0, 56.0),
        ]

        # Away Team (Red, defending right goal at x=105)
        away_players = [
            Player("A1", "away", 1, "Courtois", "GK", 101.5, 34.0),
            Player("A2", "away", 2, "Carvajal", "RB", 80.0, 56.0),
            Player("A3", "away", 3, "Davies", "LB", 76.0, 12.0),
            Player("A4", "away", 4, "Rudiger", "CB", 87.0, 28.0),
            Player("A5", "away", 5, "Militao", "CB", 88.0, 40.0),
            Player("A6", "away", 6, "Rice", "CM", 65.0, 32.0),
            Player("A7", "away", 7, "Mbappe", "RW", 48.0, 55.0),
            Player("A8", "away", 8, "Bellingham", "CM", 62.0, 42.0),
            Player("A9", "away", 9, "Kane", "ST", 40.0, 34.0),
            Player("A10", "away", 10, "Modric", "CAM", 56.0, 36.0),
            Player("A11", "away", 11, "Son", "LW", 46.0, 14.0),
        ]

        for p in home_players + away_players:
            state.players[p.id] = p

        return state

    @staticmethod
    def compute_lane_clearance(state: MatchState, from_pos: Tuple[float, float], to_pos: Tuple[float, float], defending_team: str) -> float:
        """Calculates percentage of clear passing lane vs defenders (100% = open, 0% = blocked)."""
        fx, fy = from_pos
        tx, ty = to_pos
        dx = tx - fx
        dy = ty - fy
        dist = math.hypot(dx, dy)
        if dist < 0.1:
            return 1.0

        ux = dx / dist
        uy = dy / dist

        min_dist_to_line = 999.0
        for p in state.get_team_players(defending_team):
            # Projection of defender onto pass segment
            vx = p.x - fx
            vy = p.y - fy
            proj = vx * ux + vy * uy
            if 0.5 <= proj <= dist - 0.5:
                # Perpendicular distance to line
                perp = abs(vx * uy - vy * ux)
                if perp < min_dist_to_line:
                    min_dist_to_line = perp

        # Clearance is 100% if nearest defender is >= 3.0m away from trajectory
        if min_dist_to_line >= 3.0:
            return 1.0
        elif min_dist_to_line <= 0.6:
            return 0.15
        else:
            return max(0.15, min_dist_to_line / 3.0)

    @staticmethod
    def step_kinematics(state: MatchState, dt: float = 0.25):
        """Advances positions of players and ball according to velocity vectors."""
        state.second += dt
        if state.second >= 60.0:
            state.second -= 60.0
            state.minute += 1

        # Integrate ball
        b = state.ball
        b.x += b.vx * dt
        b.y += b.vy * dt
        b.vx *= 0.92  # Grass drag
        b.vy *= 0.92

        # Clamp pitch boundaries
        b.x = max(0.0, min(PITCH_LENGTH, b.x))
        b.y = max(0.0, min(PITCH_WIDTH, b.y))

        # Integrate players
        for p in state.players.values():
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.x = max(0.5, min(PITCH_LENGTH - 0.5, p.x))
            p.y = max(0.5, min(PITCH_WIDTH - 0.5, p.y))
            p.vx *= 0.85
            p.vy *= 0.85
