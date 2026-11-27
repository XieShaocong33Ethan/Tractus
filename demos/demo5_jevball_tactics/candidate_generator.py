"""demos/demo5_jevball_tactics/candidate_generator.py

Generates up to 14 typed tactical candidate actions per active player,
extracting exact geometric and physical features (lane clearance, xG, ETA threat, offside margin).
Strictly matches the JevBall System One decision contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .pitch_sim import MatchState, PITCH_LENGTH, PITCH_WIDTH, PitchSimulator, Player


@dataclass
class TacticalCandidate:
    """Represents a discrete candidate action evaluated by the neural model."""
    slot: str  # 'A', 'B', 'C', 'D', 'E', ...
    action_id: str
    label: str
    category: str  # 'PASS', 'DRIBBLE', 'SHOT', 'DEFENSE', 'HOLD'
    distance: float  # meters
    lane_clearance: float  # 0.0 ~ 1.0 (1.0 = clear line)
    receiver_space: float  # meters to nearest defender
    success_est: float  # 0.0 ~ 1.0
    xg: float  # expected goals
    threat_eta: float  # seconds until nearest defender can tackle
    offside_margin: float  # negative = onside, positive = offside
    target_pos: Tuple[float, float]
    target_player_id: Optional[str] = None
    risk_level: str = "LOW"  # 'LOW', 'MEDIUM', 'HIGH'

    @property
    def compact_fact_string(self) -> str:
        """Compact string formatted for JEV System One prompt token efficiency."""
        parts = [f"dist: {self.distance:.0f}m"]
        if self.category == "PASS":
            parts.append(f"clearance: {self.lane_clearance*100:.0f}%")
            parts.append(f"space: {self.receiver_space:.1f}m")
            if self.offside_margin > -90:
                onside_tag = "onside" if self.offside_margin <= 0 else "OFFSIDE"
                parts.append(f"{onside_tag} ({self.offside_margin:+.1f}m)")
            if self.xg > 0.05:
                parts.append(f"xG: {self.xg:.2f}")
        elif self.category == "SHOT":
            parts.append(f"xG: {self.xg:.2f}")
            parts.append(f"clearance: {self.lane_clearance*100:.0f}%")
        elif self.category == "DRIBBLE":
            parts.append(f"threat ETA: {self.threat_eta:.2f}s")
            parts.append(f"success: {self.success_est*100:.0f}%")
        return ", ".join(parts)


class TacticalCandidateGenerator:
    """Extracts realistic tactical candidate moves for the ball carrier."""

    SLOTS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]

    @classmethod
    def generate_candidates(cls, state: MatchState, carrier: Optional[Player] = None) -> List[TacticalCandidate]:
        """Generates realistic tactical choices for the current carrier."""
        if carrier is None:
            carrier = state.active_carrier
        if carrier is None:
            return []

        team = carrier.team
        opp_team = "away" if team == "home" else "home"
        target_goal_x = PITCH_LENGTH if team == "home" else 0.0
        target_goal_y = PITCH_WIDTH / 2.0
        offside_x = state.offside_line_x(opp_team)

        # Nearest opponent threat calculation
        opponents = state.get_team_players(opp_team)
        min_opp_dist = 999.0
        for opp in opponents:
            d = carrier.dist_to(opp.x, opp.y)
            if d < min_opp_dist:
                min_opp_dist = d
        threat_eta = max(0.2, min_opp_dist / 6.5)  # assuming defender closes at ~6.5m/s

        candidates: List[TacticalCandidate] = []

        # 1. Evaluate Teammates for Pass Options
        teammates = [p for p in state.get_team_players(team) if p.id != carrier.id]
        
        # Sort teammates by forward progression
        def forward_val(p: Player) -> float:
            return p.x if team == "home" else (PITCH_LENGTH - p.x)
        sorted_teammates = sorted(teammates, key=forward_val, reverse=True)

        for mate in sorted_teammates:
            dist = carrier.dist_to(mate.x, mate.y)
            if dist < 2.0 or dist > 55.0:
                continue

            clearance = PitchSimulator.compute_lane_clearance(state, carrier.pos, mate.pos, opp_team)
            
            # Distance of nearest opponent to receiver
            min_space_to_mate = min([mate.dist_to(opp.x, opp.y) for opp in opponents], default=10.0)
            
            # Offside check
            offside_margin = (mate.x - offside_x) if team == "home" else (offside_x - mate.x)
            is_offside = (offside_margin > 0.3) and (mate.x > 52.5 if team == "home" else mate.x < 52.5)

            # Heuristic xG contribution
            goal_dist = math.hypot(target_goal_x - mate.x, target_goal_y - mate.y)
            mate_xg = max(0.01, min(0.75, math.exp(-goal_dist / 14.0)))

            if is_offside:
                continue  # Never offer obviously offside passes

            # Through pass condition: receiver running behind defensive line but onside
            if mate.role in ["ST", "LW", "RW"] and mate.x >= 70.0 and offside_margin >= -3.5:
                # Through pass into space in front of runner
                lead_x = min(PITCH_LENGTH - 6.0, mate.x + 8.5)
                lead_y = mate.y + (0.5 if mate.y > 34 else -0.5)
                cand = TacticalCandidate(
                    slot="",
                    action_id=f"through_{mate.id.lower()}",
                    label=f"Through Ball into channel to #{mate.number} {mate.name}",
                    category="PASS",
                    distance=carrier.dist_to(lead_x, lead_y),
                    lane_clearance=clearance,
                    receiver_space=min_space_to_mate,
                    success_est=max(0.3, clearance * 0.95),
                    xg=round(mate_xg * 1.35, 2),
                    threat_eta=threat_eta,
                    offside_margin=round(offside_margin, 2),
                    target_pos=(lead_x, lead_y),
                    target_player_id=mate.id,
                    risk_level="MEDIUM" if clearance > 0.8 else "HIGH",
                )
                candidates.append(cand)
            elif clearance >= 0.5:
                # Regular Pass to feet
                cand = TacticalCandidate(
                    slot="",
                    action_id=f"pass_{mate.id.lower()}",
                    label=f"Pass to feet of #{mate.number} {mate.name} ({mate.role})",
                    category="PASS",
                    distance=dist,
                    lane_clearance=clearance,
                    receiver_space=min_space_to_mate,
                    success_est=clearance * 0.98,
                    xg=round(mate_xg, 2),
                    threat_eta=threat_eta,
                    offside_margin=round(offside_margin, 2),
                    target_pos=mate.pos,
                    target_player_id=mate.id,
                    risk_level="LOW" if clearance >= 0.85 else "MEDIUM",
                )
                candidates.append(cand)

        # 2. Evaluate Shot Options if within 35m of goal
        dist_to_goal = math.hypot(target_goal_x - carrier.x, target_goal_y - carrier.y)
        if dist_to_goal <= 35.0:
            direct_clearance = PitchSimulator.compute_lane_clearance(state, carrier.pos, (target_goal_x, target_goal_y), opp_team)
            base_xg = max(0.03, min(0.85, math.exp(-dist_to_goal / 12.0) * direct_clearance))
            
            if dist_to_goal <= 20.0:
                candidates.append(TacticalCandidate(
                    slot="",
                    action_id="shoot_clinical",
                    label=f"Clinical Low Drive to bottom corner (dist: {dist_to_goal:.1f}m)",
                    category="SHOT",
                    distance=dist_to_goal,
                    lane_clearance=direct_clearance,
                    receiver_space=0.0,
                    success_est=direct_clearance * 0.9,
                    xg=round(base_xg * 1.2, 2),
                    threat_eta=threat_eta,
                    offside_margin=-99.0,
                    target_pos=(target_goal_x, target_goal_y),
                    risk_level="LOW",
                ))
            else:
                candidates.append(TacticalCandidate(
                    slot="",
                    action_id="shoot_curled",
                    label=f"Curled Finesse Shot towards far post (dist: {dist_to_goal:.1f}m)",
                    category="SHOT",
                    distance=dist_to_goal,
                    lane_clearance=direct_clearance,
                    receiver_space=0.0,
                    success_est=direct_clearance * 0.75,
                    xg=round(base_xg, 2),
                    threat_eta=threat_eta,
                    offside_margin=-99.0,
                    target_pos=(target_goal_x, target_goal_y),
                    risk_level="MEDIUM",
                ))

        # 3. Solo Dribble Forward
        dribble_lead_x = min(PITCH_LENGTH - 2.0, carrier.x + 6.0)
        dribble_lead_y = carrier.y
        dribble_clearance = PitchSimulator.compute_lane_clearance(state, carrier.pos, (dribble_lead_x, dribble_lead_y), opp_team)
        candidates.append(TacticalCandidate(
            slot="",
            action_id="dribble_fwd",
            label="Solo Dribble forward attacking space",
            category="DRIBBLE",
            distance=6.0,
            lane_clearance=dribble_clearance,
            receiver_space=min_opp_dist,
            success_est=max(0.2, min_opp_dist / 4.0),
            xg=0.08,
            threat_eta=threat_eta,
            offside_margin=-99.0,
            target_pos=(dribble_lead_x, dribble_lead_y),
            risk_level="HIGH" if threat_eta < 0.8 else "MEDIUM",
        ))

        # 4. Safe Reset Backpass
        cm_mates = [p for p in teammates if p.role in ["CM", "CB"] and (p.x < carrier.x if team == "home" else p.x > carrier.x)]
        if cm_mates:
            safe_mate = sorted(cm_mates, key=lambda p: carrier.dist_to(p.x, p.y))[0]
            dist = carrier.dist_to(safe_mate.x, safe_mate.y)
            clearance = PitchSimulator.compute_lane_clearance(state, carrier.pos, safe_mate.pos, opp_team)
            candidates.append(TacticalCandidate(
                slot="",
                action_id=f"pass_safe_{safe_mate.id.lower()}",
                label=f"Safe Lateral Backpass to #{safe_mate.number} {safe_mate.name} to recycle tempo",
                category="PASS",
                distance=dist,
                lane_clearance=clearance,
                receiver_space=min([safe_mate.dist_to(opp.x, opp.y) for opp in opponents], default=10.0),
                success_est=0.98,
                xg=0.01,
                threat_eta=threat_eta,
                offside_margin=-99.0,
                target_pos=safe_mate.pos,
                target_player_id=safe_mate.id,
                risk_level="LOW",
            ))

        # Sort and limit to at most 14 candidates (JEV contract limit)
        # Prioritize through balls, shots, high-clearance passes, and dribble
        def priority(c: TacticalCandidate) -> float:
            score = 0.0
            if "through" in c.action_id:
                score += 50.0 + c.xg * 30.0
            elif c.category == "SHOT":
                score += 40.0 + c.xg * 40.0
            elif "dribble" in c.action_id:
                score += 20.0
            elif "pass_safe" in c.action_id:
                score += 15.0
            else:
                score += 25.0 + c.lane_clearance * 10.0
            return score

        candidates.sort(key=priority, reverse=True)
        final_candidates = candidates[:14]

        # Assign slot letters A, B, C, ...
        for idx, cand in enumerate(final_candidates):
            cand.slot = cls.SLOTS[idx]

        return final_candidates
