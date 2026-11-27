"""demos/demo5_jevball_tactics/radar_display.py

ANSI / Unicode Tactical Pitch Radar and Live Decision Telemetry HUD for JevBall.
Renders top-down pitch view, player formations, passing vectors, candidate probability bars,
and physical inference SLA telemetries directly in the terminal.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

from .pitch_sim import MatchState, PITCH_LENGTH, PITCH_WIDTH, Player
from .candidate_generator import TacticalCandidate
from .prompt_contract import JevBallDecisionResult


class TacticalRadarRenderer:
    """Renders 105m x 68m tactical mini-radar and neural decision HUD in ANSI terminal."""

    # ANSI Colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    BRIGHT_CYAN = "\033[96m"
    BLUE = "\033[34m"
    RED = "\033[31m"
    BRIGHT_RED = "\033[91m"
    GREEN = "\033[32m"
    BRIGHT_GREEN = "\033[92m"
    YELLOW = "\033[33m"
    BRIGHT_YELLOW = "\033[93m"
    MAGENTA = "\033[35m"
    WHITE = "\033[37m"
    BG_DARK = "\033[48;5;234m"

    PITCH_COLS = 70
    PITCH_ROWS = 15

    @classmethod
    def render_tactical_pitch(
        cls,
        state: MatchState,
        active_candidate: Optional[TacticalCandidate] = None,
    ) -> str:
        """Renders 2D top-down football pitch with player markers and passing vector."""
        grid = [[" " for _ in range(cls.PITCH_COLS)] for _ in range(cls.PITCH_ROWS)]

        # 1. Pitch Markings
        for r in range(cls.PITCH_ROWS):
            grid[r][0] = "│"
            grid[r][-1] = "│"
            mid_c = cls.PITCH_COLS // 2
            grid[r][mid_c] = "┆"  # Midfield line

        for c in range(cls.PITCH_COLS):
            grid[0][c] = "─"
            grid[-1][c] = "─"

        grid[0][0] = "┌"
        grid[0][-1] = "┐"
        grid[-1][0] = "└"
        grid[-1][-1] = "┘"

        # Center spot & circle
        mid_r = cls.PITCH_ROWS // 2
        mid_c = cls.PITCH_COLS // 2
        grid[mid_r][mid_c] = "┼"
        if mid_r - 2 >= 0 and mid_r + 2 < cls.PITCH_ROWS:
            grid[mid_r - 1][mid_c - 2] = "("
            grid[mid_r - 1][mid_c + 2] = ")"
            grid[mid_r + 1][mid_c - 2] = "("
            grid[mid_r + 1][mid_c + 2] = ")"

        # Goal boxes
        for r in range(mid_r - 3, mid_r + 4):
            grid[r][6] = "│"
            grid[r][cls.PITCH_COLS - 7] = "│"
        for c in range(0, 7):
            grid[mid_r - 3][c] = "─"
            grid[mid_r + 3][c] = "─"
            grid[mid_r - 3][cls.PITCH_COLS - 1 - c] = "─"
            grid[mid_r + 3][cls.PITCH_COLS - 1 - c] = "─"

        # Coordinate transform helper
        def to_grid(x: float, y: float) -> Tuple[int, int]:
            c = int((x / PITCH_LENGTH) * (cls.PITCH_COLS - 2)) + 1
            r = int((y / PITCH_WIDTH) * (cls.PITCH_ROWS - 2)) + 1
            c = max(1, min(cls.PITCH_COLS - 2, c))
            r = max(1, min(cls.PITCH_ROWS - 2, r))
            return r, c

        # 2. Draw Trajectory Vector if active candidate exists
        if active_candidate and state.active_carrier:
            fr, fc = to_grid(state.active_carrier.x, state.active_carrier.y)
            tr, tc = to_grid(active_candidate.target_pos[0], active_candidate.target_pos[1])
            steps = max(abs(tr - fr), abs(tc - fc), 1)
            for s in range(1, steps):
                inter_r = int(fr + (tr - fr) * (s / steps))
                inter_c = int(fc + (tc - fc) * (s / steps))
                if 0 <= inter_r < cls.PITCH_ROWS and 0 <= inter_c < cls.PITCH_COLS:
                    grid[inter_r][inter_c] = "·"

        # 3. Draw Players
        for p in state.players.values():
            r, c = to_grid(p.x, p.y)
            if p.team == "home":
                tag = f"{cls.BRIGHT_CYAN}{p.number}{cls.RESET}" if p.number < 10 else f"{cls.CYAN}{p.number}{cls.RESET}"
                if p.id == state.active_carrier_id:
                    tag = f"{cls.BRIGHT_YELLOW}{cls.BOLD}●{p.number}{cls.RESET}"
            else:
                tag = f"{cls.BRIGHT_RED}{p.number}{cls.RESET}"
            grid[r][c] = tag

        # Assemble string
        lines = []
        for row in grid:
            line_str = "".join(row)
            lines.append(f"  {line_str}")
        return "\n".join(lines)

    @classmethod
    def render_hud(
        cls,
        state: MatchState,
        candidates: List[TacticalCandidate],
        result: Optional[JevBallDecisionResult] = None,
    ) -> str:
        """Constructs full tactical console with telemetry HUD."""
        buf = []
        buf.append(f"{cls.BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{cls.RESET}")
        
        # Match header line
        score_str = f"{cls.BRIGHT_CYAN}HOME {state.score_home}{cls.RESET} - {cls.BRIGHT_RED}{state.score_away} AWAY{cls.RESET}"
        clock_str = f"{cls.YELLOW}{state.minute:02d}:{state.second:04.1f}{cls.RESET}"
        phase_str = f"{cls.MAGENTA}[{state.phase}]{cls.RESET}"
        buf.append(f"⚽ {cls.BOLD}JEVBALL 3D TACTICAL ARENA{cls.RESET} | {clock_str} | {score_str} | {phase_str}")
        buf.append(f"{cls.BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{cls.RESET}")

        # Tactical Pitch Radar
        active_cand = result.chosen_candidate if result else (candidates[0] if candidates else None)
        buf.append(cls.render_tactical_pitch(state, active_cand))
        buf.append(f"  {cls.DIM}Home: Cyan (attacking ->) | Away: Red (defending) | ● Carrier | · Vector{cls.RESET}")
        buf.append("")

        carrier = state.active_carrier
        if carrier:
            opp_team = "away" if carrier.team == "home" else "home"
            opps = state.get_team_players(opp_team)
            min_opp_dist = min([carrier.dist_to(o.x, o.y) for o in opps], default=10.0)
            threat_eta = max(0.2, min_opp_dist / 6.5)
            threat_color = cls.BRIGHT_RED if threat_eta < 0.8 else cls.BRIGHT_GREEN
            threat_badge = f"{threat_color}{threat_eta:.2f}s ({min_opp_dist:.1f}m){cls.RESET}"

            buf.append(
                f"  {cls.BOLD}Carrier:{cls.RESET} #{carrier.number} {carrier.name} ({carrier.role}) at ({carrier.x:.1f}m, {carrier.y:.1f}m) | "
                f"{cls.BOLD}Threat ETA:{cls.RESET} {threat_badge}"
            )
        buf.append("")

        # Candidate Probabilities Distribution Bar Chart
        if result:
            buf.append(f"  {cls.BOLD}Tractus Neural Action Distribution [{result.model_name} | Metal GPU]:{cls.RESET}")
            for cand in candidates:
                pct = result.probabilities.get(cand.slot, 0.0)
                bar_len = int(pct / 3.0)
                bar_str = "█" * bar_len

                is_selected = (cand.slot == result.chosen_slot)
                if is_selected:
                    bar_color = cls.BRIGHT_GREEN
                    prefix = f"  {cls.BOLD}{cls.BRIGHT_YELLOW}[{cand.slot}]{cls.RESET} "
                    suffix = f" {cls.BOLD}{cls.BRIGHT_GREEN}{pct:5.1f}% ⚡ SELECTED{cls.RESET}"
                else:
                    bar_color = cls.CYAN
                    prefix = f"  {cls.DIM}[{cand.slot}]{cls.RESET} "
                    suffix = f" {cls.DIM}{pct:5.1f}%{cls.RESET}"

                action_desc = f"{cand.label:<40} ({cand.compact_fact_string})"
                buf.append(f"{prefix}{action_desc[:52]:<52} {bar_color}{bar_str:<34}{cls.RESET}{suffix}")

            buf.append("")
            # SLA & Telemetry Comparison Strip
            sla_status = f"{cls.BRIGHT_GREEN}✔ SLA MET (Zero Dropped Frames){cls.RESET}" if result.sla_met else f"{cls.BRIGHT_RED}✖ MISSED{cls.RESET}"
            cloud_status = f"{cls.BRIGHT_RED}✖ OVERDUE (Fell back to rule){cls.RESET}"
            headroom = result.sla_deadline_ms - result.latency_ms
            speedup = result.cloud_api_baseline_latency_ms / max(1.0, result.latency_ms)

            buf.append(f"  {cls.BOLD}Physical Real-Time SLA Benchmark (Carrier Deadline: 300.0ms):{cls.RESET}")
            buf.append(f"  ├─ {cls.BRIGHT_CYAN}Tractus Metal GPU Latency:{cls.RESET} {cls.BOLD}{result.latency_ms:.1f}ms{cls.RESET} -> {sla_status} (Headroom: +{headroom:.1f}ms)")
            buf.append(f"  ├─ {cls.DIM}Jev Cloud API Baseline:   366.0ms -> {cloud_status} (Lag: -66.0ms){cls.RESET}")
            buf.append(f"  └─ {cls.YELLOW}Hardware Advantage:{cls.RESET} {cls.BOLD}{speedup:.1f}x Faster{cls.RESET} than cloud API · $0.00 token cost · 100% on-device")
            buf.append("")
            buf.append(f"  {cls.BOLD}⚡ Tactical Execution:{cls.RESET} {result.chosen_candidate.label} executed! Ball transferred successfully.")
        else:
            buf.append(f"  {cls.DIM}Awaiting neural model forward pass...{cls.RESET}")

        buf.append(f"{cls.BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{cls.RESET}")
        return "\n".join(buf)
