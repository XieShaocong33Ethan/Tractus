"""
Millisecond-Precision Physical Stopwatch HUD (Head-Up Display).

Provides millisecond-accurate stopwatch telemetry with dynamic ANSI color states.

Features:
- Sub-millisecond timing accuracy (00:00.000 or 0.024s)
- Real-time color state alerts (Green running, Red slow baseline, Bold locked)
- Comparative side-by-side benchmarking (Tractus Fast-Path vs Traditional LLM)
"""

import time
from enum import Enum
from typing import Optional


class HUDState(Enum):
    """HUD state enumeration."""
    IDLE = "idle"           # Idle state
    RUNNING = "running"     # Active inference (green timer)
    LOCKED = "locked"       # Decision locked (completion banner)
    SLOW = "slow"           # Slow comparison baseline (red timer)


class StopwatchHUD:
    """Millisecond-precision stopwatch HUD component."""

    def __init__(self, label: str = "Tractus"):
        """
        Initializes stopwatch timer.

        Args:
            label: Display label (e.g. "Tractus", "Baseline LLM")
        """
        self.label = label
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.state = HUDState.IDLE
        self.locked_duration: Optional[float] = None

    def start(self, is_slow: bool = False):
        """
        Starts stopwatch.

        Args:
            is_slow: Whether this timer represents a slow comparison baseline.
        """
        self.start_time = time.perf_counter()
        self.end_time = None
        self.locked_duration = None
        self.state = HUDState.SLOW if is_slow else HUDState.RUNNING

    def stop(self) -> float:
        """
        Stops stopwatch and locks display duration.

        Returns:
            Elapsed time in seconds.
        """
        if self.start_time is None:
            raise RuntimeError("Timer not started")

        self.end_time = time.perf_counter()
        self.locked_duration = self.end_time - self.start_time
        self.state = HUDState.LOCKED
        return self.locked_duration

    def get_elapsed(self) -> float:
        """
        Gets current elapsed time in seconds.

        Returns:
            Elapsed seconds or 0.0 if not started.
        """
        if self.start_time is None:
            return 0.0

        if self.locked_duration is not None:
            return self.locked_duration

        return time.perf_counter() - self.start_time

    def format_time(self, seconds: float) -> str:
        """
        Formats seconds into MM:SS.mmm or SS.mmm format.

        Args:
            seconds: Time duration in seconds.

        Returns:
            Formatted time string.
        """
        if seconds < 1.0:
            return f"{seconds:.3f}s"
        elif seconds < 60.0:
            return f"{seconds:.3f}s"
        else:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes:02d}:{remaining_seconds:06.3f}"

    def render(self) -> str:
        """
        Renders colored HUD display text.

        Returns:
            ANSI-formatted string.
        """
        elapsed = self.get_elapsed()
        time_str = self.format_time(elapsed)

        # ANSI color codes
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        if self.state == HUDState.IDLE:
            return f"{YELLOW}[{self.label}]{RESET} {time_str}"
        elif self.state == HUDState.RUNNING:
            return f"{GREEN}{BOLD}[{self.label}]{RESET} {GREEN}{time_str}{RESET}"
        elif self.state == HUDState.SLOW:
            return f"{RED}{BOLD}[{self.label}]{RESET} {RED}{time_str}{RESET}"
        elif self.state == HUDState.LOCKED:
            return f"{BOLD}[{self.label}]{RESET} ✓ {BOLD}{time_str}{RESET}"

        return f"[{self.label}] {time_str}"

    def reset(self):
        """Resets stopwatch state."""
        self.start_time = None
        self.end_time = None
        self.state = HUDState.IDLE
        self.locked_duration = None


class ComparativeHUD:
    """Side-by-side comparative HUD (Tractus vs Baseline LLM)."""

    def __init__(self):
        self.tractus_hud = StopwatchHUD(label="Tractus")
        self.baseline_hud = StopwatchHUD(label="Baseline LLM")

    def start_tractus(self):
        """Starts Tractus fast timer."""
        self.tractus_hud.start(is_slow=False)

    def start_baseline(self):
        """Starts slow baseline timer."""
        self.baseline_hud.start(is_slow=True)

    def stop_tractus(self) -> float:
        """Stops Tractus timer."""
        return self.tractus_hud.stop()

    def stop_baseline(self) -> float:
        """Stops baseline timer."""
        return self.baseline_hud.stop()

    def render_side_by_side(self) -> str:
        """
        Renders formatted side-by-side comparison string.

        Returns:
            Formatted comparison text.
        """
        tractus_display = self.tractus_hud.render()
        baseline_display = self.baseline_hud.render()

        speedup = self._calculate_speedup()
        speedup_str = f" ({speedup:.1f}x faster)" if speedup > 1.0 else ""

        return f"{tractus_display}  |  {baseline_display}{speedup_str}"

    def _calculate_speedup(self) -> float:
        """Computes speedup factor."""
        tractus_time = self.tractus_hud.get_elapsed()
        baseline_time = self.baseline_hud.get_elapsed()

        if tractus_time > 0 and baseline_time > 0:
            return baseline_time / tractus_time
        return 1.0

    def reset(self):
        """Resets both timers."""
        self.tractus_hud.reset()
        self.baseline_hud.reset()
