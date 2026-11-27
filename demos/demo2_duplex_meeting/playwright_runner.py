"""demos/demo2_duplex_meeting/playwright_runner.py

Physical Native Playwright Controller for Tractus Demo 2 (Duplex Meeting Copilot).
Complies with Option B: Launches a real, native Google Chrome / Chromium desktop
window on macOS, executing authentic Zoom/Teams-grade meeting room orchestration,
AMI Meeting Corpus streaming, and full-duplex speech-act filtering & background bot
dispatching via Apple Silicon Metal GPU (MLX INT4), with full synchronized audio muxing.

Usage:
  python demos/demo2_duplex_meeting/playwright_runner.py --headless false --speed 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demos.demo2_duplex_meeting.duplex_engine import DuplexEngine, DecisionAction, IntentType
from demos.demo2_duplex_meeting.stream_simulator import StreamSimulator

logger = logging.getLogger("TractusMeetingPlaywright")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _build_demo2_synchronized_audio(chunks: List[Any], speed: float = 1.0, initial_offset_s: float = 0.5) -> Optional[Path]:
    """Constructs a frame-accurate AAC audio track matching the video timeline."""
    ffmpeg_bin = "/opt/homebrew/bin/ffmpeg" if Path("/opt/homebrew/bin/ffmpeg").exists() else "ffmpeg"
    audio_dir = PROJECT_ROOT / "demos" / "web_ui" / "static" / "audio"
    tmp_dir = Path("/tmp/demo2_audio_mux_prod")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        silence_init = tmp_dir / "silence_init.wav"
        silence_gap = tmp_dir / "silence_gap.wav"
        gap_s = 0.15 / speed
        init_s = initial_offset_s / speed

        subprocess.run(
            [ffmpeg_bin, "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", f"{init_s:.3f}", str(silence_init)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [ffmpeg_bin, "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", f"{gap_s:.3f}", str(silence_gap)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        concat_file = tmp_dir / "concat.txt"
        with open(concat_file, "w") as f:
            f.write(f"file '{silence_init.resolve()}'\n")
            for i in range(1, len(chunks) + 1):
                mp3_path = audio_dir / f"demo2_chunk_{i:02d}.mp3"
                if not mp3_path.exists():
                    continue
                wav_path = tmp_dir / f"chunk_{i:02d}.wav"
                if abs(speed - 1.0) < 0.05:
                    subprocess.run(
                        [ffmpeg_bin, "-y", "-i", str(mp3_path), "-ar", "44100", "-ac", "2", str(wav_path)],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                else:
                    subprocess.run(
                        [ffmpeg_bin, "-y", "-i", str(mp3_path), "-filter:a", f"atempo={speed}", "-ar", "44100", "-ac", "2", str(wav_path)],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                f.write(f"file '{wav_path.resolve()}'\n")
                if i < len(chunks):
                    f.write(f"file '{silence_gap.resolve()}'\n")

        out_audio = tmp_dir / "demo2_synced_master.m4a"
        subprocess.run(
            [ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c:a", "aac", "-b:a", "192k", str(out_audio)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return out_audio
    except Exception as e:
        logger.warning("Failed to build synchronized audio track: %s", e)
        return None


def _convert_webm_to_mp4(webm_path: Path, mp4_path: Path, audio_path: Optional[Path] = None) -> bool:
    """Transcodes Playwright WebM recording to 1080p H.264 MP4 with synchronized AAC audio."""
    if not webm_path.exists():
        return False
    try:
        mp4_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_bin = "/opt/homebrew/bin/ffmpeg" if Path("/opt/homebrew/bin/ffmpeg").exists() else "ffmpeg"
        if audio_path and audio_path.exists():
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i", str(webm_path),
                "-i", str(audio_path),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-r", "30",
                "-c:a", "aac",
                "-b:a", "192k",
                str(mp4_path),
            ]
        else:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i", str(webm_path),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-r", "30",
                str(mp4_path),
            ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        logger.info("✔ Transcoded native Chrome recording to MP4 (audio=%s): %s", "AAC" if audio_path else "None", mp4_path)
        return True
    except Exception as e:
        logger.warning("Failed to transcode webm to mp4: %s", e)
        return False


class TractusMeetingPlaywrightRunner:
    """Orchestrates physical native Chrome automation for Demo 2 Duplex Meeting Copilot."""

    DEFAULT_URL = "http://127.0.0.1:8080/static/apps/meeting_stage.html"

    def __init__(
        self,
        base_url: Optional[str] = None,
        headless: bool = False,
        engine_type: str = "mlx",
        model_size: str = "0.8b",
        quant_mode: str = "int4",
        speed: float = 1.0,
        output_video_path: Optional[Path] = None,
        checkpoint_path: Optional[str] = None,
    ):
        self.base_url = base_url or self.DEFAULT_URL
        self.headless = headless
        self.engine_type = engine_type
        self.model_size = model_size
        self.quant_mode = quant_mode
        self.speed = max(0.2, speed)
        self.checkpoint_path = checkpoint_path
        self.output_video_path = output_video_path or Path(
            "artifacts/videos/tractus_product_demos/02_tractus_meeting_native_chrome.mp4"
        )

        # Fallback to local file if server is offline
        if "127.0.0.1" in self.base_url:
            local_file = PROJECT_ROOT / "demos" / "web_ui" / "static" / "apps" / "meeting_stage.html"
            try:
                import urllib.request
                urllib.request.urlopen(self.base_url, timeout=0.5)
            except Exception:
                self.base_url = f"file://{local_file.resolve()}"

        # Initialize physical duplex decision engine (wraps MLX INT4 / PyTorch)
        self.duplex_engine = DuplexEngine(
            backend=self.engine_type,
            model_size=self.model_size,
            quant_mode=self.quant_mode,
            checkpoint_path=self.checkpoint_path,
        )
        self.engine = self.duplex_engine.engine

        logger.info(
            "✔ TractusMeetingPlaywrightRunner ready [Engine: %s | Model: %s | Quant: %s | URL: %s | Output: %s]",
            self.engine_type.upper(),
            self.model_size.upper(),
            self.quant_mode.upper(),
            self.base_url,
            self.output_video_path,
        )

    def run(self, keep_open_seconds: float = 3.0) -> Dict[str, Any]:
        """Launches native Chrome and processes streaming conference speech chunks."""
        from playwright.sync_api import sync_playwright

        logger.info("🚀 Launching Native Chrome Desktop Window (headless=%s, speed=%.1fx)...", self.headless, self.speed)
        sim = StreamSimulator()
        chunks = sim.meeting_transcript

        results: List[Dict[str, Any]] = []
        total_latency = 0.0
        filler_count = 0
        actionable_count = 0

        video_dir = Path("artifacts/videos/tractus_product_demos/native_chrome_videos")
        video_dir.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=self.headless,
                slow_mo=int(100 / self.speed) if not self.headless else 0,
                args=[
                    "--window-size=1280,860",
                    "--window-position=120,60",
                    "--autoplay-policy=no-user-gesture-required",
                    "--disable-features=PreloadMediaEngagementData,MediaEngagementBypassAutoplayPolicies",
                    "--no-sandbox",
                ],
            )
            context = browser.new_context(
                viewport={"width": 1280, "height": 860},
                record_video_dir=str(video_dir),
                record_video_size={"width": 1280, "height": 860},
            )
            page = context.new_page()

            logger.info("🌐 Navigating to authentic Meeting Stage application: %s", self.base_url)
            page.goto(self.base_url)
            page.wait_for_load_state("networkidle")
            time.sleep(0.5 / self.speed)

            # Process chunks through physical neural model & DuplexEngine
            for idx, chunk in enumerate(chunks, 1):
                spk_name = chunk.speaker.name if hasattr(chunk.speaker, "name") else str(chunk.speaker)
                logger.info("🎙 [Chunk %02d/%02d] Speaker %s: \"%s\"...", idx, len(chunks), spk_name, chunk.text[:35])

                # Physical MLX INT4 forward pass inside duplex_engine
                detected = self.duplex_engine.process_chunk(chunk)
                intent = self.duplex_engine.detected_intents[-1]
                total_latency += intent.neural_latency_ms

                is_silent = (intent.decision_action == DecisionAction.SILENT_OBSERVE)
                if is_silent:
                    filler_count += 1
                    logger.info("   -> [SILENT OBSERVE] Local Filtered (%s). Latency: %.2fms | Saved: %d tokens",
                                intent.intent_type.value, intent.neural_latency_ms, intent.token_saved)
                else:
                    actionable_count += 1
                    logger.info("   -> [ACTIVE CAPTURED] Action: %s | Latency: %.2fms",
                                intent.decision_action.value, intent.neural_latency_ms)

                # Prepare task payload for autonomous background bot dock
                task_payload = None
                active_task = self.duplex_engine.active_task
                if intent.decision_action == DecisionAction.DISPATCH_TASK:
                    task_payload = {
                        "action": "DISPATCH",
                        "task_id": active_task.task_id if active_task else f"TASK-{idx}",
                        "title": active_task.title if active_task else "Meeting Support Task",
                        "creator": active_task.creator if active_task else spk_name,
                        "tokens_saved": self.duplex_engine.total_tokens_saved,
                    }
                elif intent.decision_action == DecisionAction.SUPPLEMENT_TASK:
                    task_payload = {
                        "action": "SUPPLEMENT",
                        "task_id": active_task.task_id if active_task else f"TASK-{idx}",
                        "title": active_task.title if active_task else "Meeting Support Task",
                        "amendment": chunk.text,
                        "creator": spk_name,
                        "tokens_saved": self.duplex_engine.total_tokens_saved,
                    }

                # Push event to Chrome DOM with audio synchronization
                safe_text = chunk.text.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
                audio_url = getattr(chunk, "audio_url", "")
                payload_json = json.dumps(task_payload, ensure_ascii=False) if task_payload else "null"

                page.evaluate(
                    f"window.addStreamChunk('{spk_name}', '{safe_text}', {str(is_silent).lower()}, "
                    f"'{intent.decision_action.value}', {intent.neural_latency_ms:.1f}, '{audio_url}', {payload_json})"
                )

                # Smoothly move cursor to corresponding spotlight card or bot card
                lower_txt = chunk.text.lower()
                target_id = None
                if "budget" in lower_txt or "calculation" in lower_txt or "nine point nine five" in lower_txt:
                    target_id = "#kpi-budget"
                elif "lcd display" in lower_txt or "screen" in lower_txt or "twelve point" in lower_txt or "twelve and a half" in lower_txt:
                    target_id = "#kpi-bom"
                elif "fifty million" in lower_txt or "profit" in lower_txt:
                    target_id = "#kpi-breakeven"
                elif "negotiate" in lower_txt or "vendors" in lower_txt or "save some money" in lower_txt:
                    target_id = "#kpi-vendor"

                if target_id:
                    try:
                        card_box = page.locator(target_id).bounding_box()
                        if card_box:
                            page.mouse.move(card_box["x"] + card_box["width"] / 2, card_box["y"] + card_box["height"] / 2, steps=10)
                    except Exception:
                        pass
                elif task_payload and task_payload.get("task_id"):
                    try:
                        task_box = page.locator(f"#task-card-{task_payload['task_id']}").bounding_box()
                        if task_box:
                            page.mouse.move(task_box["x"] + task_box["width"] / 2, task_box["y"] + task_box["height"] / 2, steps=10)
                    except Exception:
                        pass

                # True real-time audio playback pacing
                chunk_dur = getattr(chunk, "duration_s", 2.0)
                time.sleep(chunk_dur / self.speed)

                # Clear active speaker ring during speaker turn transition
                page.evaluate("window.clearActiveSpeaker()")
                time.sleep(0.15 / self.speed)

            # Rejection summary
            rejection_rate = (filler_count / max(1, len(chunks))) * 100.0
            page.evaluate(f"document.getElementById('val-rejection-rate').innerText = '{rejection_rate:.1f}%'")
            logger.info("🎉 Meeting Stream Processing Complete! Filler Rejection Rate: %.1f%% | Tasks: %d | Tokens Saved: %s",
                        rejection_rate, len(self.duplex_engine.dispatched_tasks), f"{self.duplex_engine.total_tokens_saved:,}")

            if keep_open_seconds > 0:
                logger.info("Holding physical Chrome window open for %.1f seconds...", keep_open_seconds)
                time.sleep(keep_open_seconds)

            video_path = page.video.path() if page.video else None
            context.close()
            browser.close()

            if video_path and Path(video_path).exists():
                synced_audio = _build_demo2_synchronized_audio(chunks, speed=self.speed, initial_offset_s=0.5)
                _convert_webm_to_mp4(Path(video_path), self.output_video_path, audio_path=synced_audio)

        summary = {
            "scenario": "meeting_stage",
            "total_chunks": len(chunks),
            "filler_count": filler_count,
            "actionable_count": actionable_count,
            "rejection_rate_pct": round(rejection_rate, 1),
            "background_tasks_dispatched": len(self.duplex_engine.dispatched_tasks),
            "total_tokens_saved": self.duplex_engine.total_tokens_saved,
            "total_neural_latency_ms": round(total_latency, 2),
            "avg_neural_latency_ms": round(total_latency / max(1, len(chunks)), 2),
            "video_path": str(self.output_video_path) if self.output_video_path.exists() else None,
            "success": True,
        }
        return summary


def main():
    parser = argparse.ArgumentParser(description="Tractus Demo 2 Native Playwright Meeting Runner")
    parser.add_argument("--url", type=str, default=None, help="Target Meeting Stage URL")
    parser.add_argument("--headless", type=str, default="false", help="Run headless ('true' or 'false')")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--engine", type=str, default="mlx", choices=["mlx", "unified"], help="Inference engine")
    parser.add_argument("--model-size", type=str, default="0.8b", help="Model size (0.8b, 2b, 4b)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom model checkpoint directory")
    parser.add_argument("--quant-mode", type=str, default="int4", choices=["int4", "fp16", "bf16"], help="Quantization mode")
    parser.add_argument("--keep-open", type=float, default=2.5, help="Seconds to keep window open after completion")
    parser.add_argument("--output", type=str, default=None, help="Output MP4 file path")

    args = parser.parse_args()
    headless_bool = args.headless.lower() == "true"
    out_path = Path(args.output) if args.output else None

    runner = TractusMeetingPlaywrightRunner(
        base_url=args.url,
        headless=headless_bool,
        engine_type=args.engine,
        model_size=args.model_size,
        checkpoint_path=args.checkpoint,
        quant_mode=args.quant_mode,
        speed=args.speed,
        output_video_path=out_path,
    )
    res = runner.run(keep_open_seconds=args.keep_open)
    print("\n" + json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
