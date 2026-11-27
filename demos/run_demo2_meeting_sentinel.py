#!/usr/bin/env python3
"""demos/run_demo_2_meeting.py

Demo 2: Real Model-Driven Duplex Meeting Copilot Runner.
Back-ended by UnifiedDecisionEngine (0.8B / 2B / 4B INT4).
Real PyTorch forward pass on each audio utterance, measuring real neural rejection latency.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root and demos directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEMOS_DIR = Path(__file__).resolve().parent
if str(DEMOS_DIR) not in sys.path:
    sys.path.insert(0, str(DEMOS_DIR))

from apse.inference.unified_decision_engine import UnifiedDecisionEngine
from demo2_duplex_meeting import StreamSimulator, DuplexEngine, IntentType
from common.stopwatch_hud import StopwatchHUD


def parse_args():
    parser = argparse.ArgumentParser(description="Demo 2: Real Model-Driven Duplex Meeting Copilot")
    parser.add_argument("--engine", type=str, default="mlx", choices=["mlx", "pytorch"], help="Inference backend (mlx for Apple Silicon Metal GPU, pytorch for CPU/CUDA)")
    parser.add_argument("--model-size", type=str, default="0.8b", choices=["0.8b", "2b", "4b", "9b"], help="Model size to evaluate")
    parser.add_argument("--quant-mode", type=str, default="int4", choices=["int4", "bf16"], help="Quantization mode (PyTorch engine)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom path or Hugging Face repo ID for model checkpoint")
    parser.add_argument("--num-chunks", type=int, default=12, help="Number of audio chunks to process (default: 12 for full Option B meeting)")
    parser.add_argument("--play-audio", action="store_true", default=True, help="Play authentic audio through system speakers (default: True on macOS)")
    parser.add_argument("--no-audio", dest="play_audio", action="store_false", help="Disable audio playback")
    return parser.parse_args()


def main():
    args = parse_args()
    print("\n" + "=" * 80)
    print(f"🎙️ Demo 2: Real Model-Driven Duplex Meeting Copilot [{args.model_size.upper()} | {args.engine.upper()}]")
    print("=" * 80)
    print("Physical Invariant: 100% Real Neural Inference, zero regex, zero fake filters.")
    print("=" * 80 + "\n")

    # 1. Initialize Real Inference Engine (MLX Metal GPU or PyTorch)
    t_load = time.perf_counter()
    if args.engine == "mlx":
        from apse.inference.mlx_decision_engine import MlxDecisionEngine
        print(f"Loading {args.model_size.upper()} into Native Apple Silicon GPU Engine (MLX Metal)...")
        engine = MlxDecisionEngine.get_instance(
            model_size=args.model_size,
            checkpoint_path=args.checkpoint,
        )
        print(f"✓ Model ready in {time.perf_counter() - t_load:.2f}s on Metal GPU.\n")
    else:
        print(f"Loading {args.model_size.upper()} ({args.quant_mode.upper()}) into Unified Decision Engine...")
        engine = UnifiedDecisionEngine.get_instance(
            model_size=args.model_size,
            quant_mode=args.quant_mode,
            checkpoint_path=args.checkpoint,
        )
        print(f"✓ Model ready in {time.perf_counter() - t_load:.2f}s.\n")

    # 2. Initialize Simulator & Duplex Engine
    simulator = StreamSimulator()
    chunks = simulator.meeting_transcript[:args.num_chunks]
    duplex_engine = DuplexEngine(engine=engine)

    # 3. Start HUD Stopwatch
    hud = StopwatchHUD(label=f"Tractus-{args.model_size.upper()} ({args.engine.upper()}) Duplex Stream Processing")
    hud.start()

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Streaming {len(chunks)} Audio Utterance Slices through Real Model...")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    for i, chunk in enumerate(chunks, 1):
        # Hardware audio output on macOS
        audio_subp = None
        if args.play_audio and sys.platform == "darwin":
            audio_path = PROJECT_ROOT / "demos" / "web_ui" / "static" / "audio" / f"demo2_chunk_{i:02d}.mp3"
            if audio_path.exists():
                import subprocess
                audio_subp = subprocess.Popen(["afplay", str(audio_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        detected = duplex_engine.process_chunk(chunk)
        intent = duplex_engine.detected_intents[-1]

        act = intent.decision_action.value
        if act == "DISPATCH_TASK":
            status = f"🚀 [DISPATCH_TASK -> LLM BOT: {intent.active_task.task_id if intent.active_task else ''}]"
        elif act == "SUPPLEMENT_TASK":
            status = f"➕ [SUPPLEMENT_TASK -> #{intent.active_task.task_id if intent.active_task else ''}]"
        elif intent.intent_type.value == "FILLER":
            status = f"🟢 [SILENT_OBSERVE: Saved {intent.token_saved} Tokens]"
        else:
            status = f"💬 [BUSINESS STATEMENT: {intent.intent_type.value}]"

        audio_tag = " 🔊 [AUDIO LIVE]" if audio_subp else ""
        print(
            f"Chunk {i:02d} | Speaker: {chunk.speaker.value:<10} | Text: \"{chunk.text}\"\n"
            f"         -> {status:<36} | Latency: {intent.neural_latency_ms:6.2f}ms | Conf: {intent.confidence*100:5.1f}%{audio_tag}"
        )
        if act == "DISPATCH_TASK" and intent.active_task:
            print(f"         📋 [TASK INITIATED] {intent.active_task.title} -> Dispatched to background agent")
        elif act == "SUPPLEMENT_TASK" and intent.active_task:
            print(f"         ➕ [AMENDMENT APPENDED] Appending parameter constraints to task #{intent.active_task.task_id} (Zero conflict)")
        elif intent.token_saved > 0:
            print(f"         🛡️ [TOKEN SHIELD] Intercepted filler speech locally, zero context overhead forwarded to backend")
        print()

        if audio_subp:
            audio_subp.wait()

    hud.stop()

    summary = duplex_engine.get_summary()

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Processing Completed!")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    print(hud.render())
    print()

    print("📊 Real Physical Metrics & Token Economy:")
    print(f"  • Total Slices Processed:   {summary['total_chunks']}")
    print(f"  • Silent / Filler Filtered: {summary['filler_count']}")
    print(f"  • Dispatched LLM Tasks:     {summary.get('dispatched_tasks_count', 0)}")
    print(f"  • Total Tokens Saved:       {summary.get('total_tokens_saved', 0):,} Tokens")
    print(f"  • Filler Rejection Rate:    {summary['filler_rejection_rate'] * 100:.1f}%")
    print(f"  • Total Neural Latency:     {summary['total_neural_latency_ms']:.1f}ms")
    print(f"  • Avg Neural Latency:       {summary['avg_neural_latency_ms']:.1f}ms per slice")
    print()


if __name__ == "__main__":
    main()
