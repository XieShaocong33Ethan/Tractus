#!/usr/bin/env python3
"""demos/run_demo_3_service.py

Demo 3: Real Model-Driven Multi-Intent Customer Service & Escalation Runner.
Back-ended by UnifiedDecisionEngine (0.8B / 2B / 4B INT4).
Performs real PyTorch forward passes to extract multi-dimensional risk slots and trigger handoff.
Zero dictionary matching, zero mock sleeps.
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
from demo3_multi_intent_service import IntentParser, HandoffRouter
from common.stopwatch_hud import StopwatchHUD


def parse_args():
    parser = argparse.ArgumentParser(description="Demo 3: Real Model Customer Service & Handoff")
    parser.add_argument("--engine", type=str, default="mlx", choices=["mlx", "pytorch"], help="Inference backend (mlx for Apple Silicon Metal GPU, pytorch for CPU/CUDA)")
    parser.add_argument("--model-size", type=str, default="0.8b", choices=["0.8b", "2b", "4b", "9b"], help="Model size to evaluate")
    parser.add_argument("--quant-mode", type=str, default="int4", choices=["int4", "bf16"], help="Quantization mode (PyTorch engine)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom path or Hugging Face repo ID for model checkpoint")
    parser.add_argument("--complaint", type=str, default="", help="Custom complaint text")
    return parser.parse_args()


def main():
    args = parse_args()
    print("\n" + "=" * 80)
    print(f"💬 Demo 3: Real Model Multi-Intent Customer Service Hub [{args.model_size.upper()} | {args.engine.upper()}]")
    print("=" * 80)
    print("Physical Invariant: 100% Real Neural Inference, zero keyword dicts, zero time.sleep.")
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

    # 2. Customer complaint text (Authentic PolyAI MINDS-14 CC-BY-4.0)
    manifest_path = PROJECT_ROOT / "demos" / "web_ui" / "static" / "audio" / "manifest.json"
    default_text = (
        "Hello, I would like to inquire about direct debit payment methods, which items are currently debited from my account, and how to cancel or refund a direct debit. "
        "Hi, my card is lost or missing, so I need to freeze my card immediately. It feels critical, so please help me complete this procedure as quickly as possible, thank you. "
        "Yes, please freeze my card right away! There are suspicious transactions on it, and I think it was stolen. Please help me process this immediately, thank you!"
    )
    if manifest_path.exists():
        try:
            import json
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            txt = data.get("demo3_complaint", {}).get("full_track", {}).get("text")
            if txt:
                default_text = txt
        except Exception:
            pass

    user_input = args.complaint.strip() or default_text

    print("📝 Customer Complaint Utterance:")
    print(f'  "{user_input}"\n')

    # 3. Initialize Parser and Router
    parser = IntentParser(engine=engine)
    router = HandoffRouter()

    hud = StopwatchHUD(label=f"Tractus-{args.model_size.upper()} Multi-Intent Neural Analysis")
    hud.start()

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Running Multi-Head Neural Intent & Risk Analysis...")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    intents = parser.parse(user_input)

    print(f"✓ Neural Model Identified {len(intents)} Intent Slots:\n")
    for i, intent in enumerate(intents, 1):
        priority_label = "🔴" if intent.priority >= 4 else "🟡"
        print(f"  {i}. {priority_label} [{intent.slot.value}] | Conf: {intent.confidence * 100:.1f}% | Priority: {intent.priority}/5")

    print()

    needs_handoff = any(i.slot.value == "transfer_human" or i.priority >= 4 for i in intents)

    if needs_handoff:
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("Triggering Human Agent Escalation & Risk Card Generation...")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

        session_id = "sess_live_neural_001"
        handoff_context = router.initiate_handoff(session_id, user_input, intents)

        print(f"✓ Escalation Dispatched (Routing Latency: {handoff_context.handoff_time_ms:.2f}ms)")
        print(f"✓ Assigned Senior Specialist Agent: {handoff_context.agent_id}\n")

        print("🎴 Live Agent Context Risk Cards:")
        for card in handoff_context.risk_cards:
            print(f"  • {card.title} [{card.risk_level.upper()}]")
            for det in card.details:
                print(f"      - {det}")
            for act in card.suggested_actions:
                print(f"      -> {act}")
            print()

    hud.stop()

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Execution Completed!")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    print(hud.render())
    print()


if __name__ == "__main__":
    main()
