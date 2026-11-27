#!/usr/bin/env python3
"""demos/run_demo_1_web.py

Demo 1: Real Model-Driven Web/OS Navigation Agent Runner.
Back-ended by UnifiedDecisionEngine on physical weights (0.8B / 2B / 4B INT4).
Real PyTorch forward pass on each navigation step, measuring real hardware latency.
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
from demo1_web_automation import MockWebApp, AutomationExecutor
from common.stopwatch_hud import StopwatchHUD


def parse_args():
    parser = argparse.ArgumentParser(description="Demo 1: Real Model-Driven Web Navigation")
    parser.add_argument("--engine", type=str, default="mlx", choices=["mlx", "pytorch"], help="Inference backend (mlx for Apple Silicon Metal GPU, pytorch for CPU/CUDA)")
    parser.add_argument("--model-size", type=str, default="0.8b", choices=["0.8b", "2b", "4b", "9b"], help="Model size to evaluate")
    parser.add_argument("--quant-mode", type=str, default="int4", choices=["int4", "bf16"], help="Quantization mode (PyTorch engine)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom path or Hugging Face repo ID for model checkpoint")
    parser.add_argument("--max-steps", type=int, default=25, help="Max autonomous steps (default: 25 for full form completion)")
    return parser.parse_args()


def main():
    args = parse_args()
    print("\n" + "=" * 80)
    print(f"🚀 Demo 1: Real Model-Driven Web/OS Navigation Agent [{args.model_size.upper()} | {args.engine.upper()}]")
    print("=" * 80)
    print("Physical Invariant: 100% Real Neural Inference, zero time.sleep, zero fake sequences.")
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

    # 2. Initialize Web App Environment
    web_app = MockWebApp()
    executor = AutomationExecutor(web_app=web_app, engine=engine)

    # 3. Start HUD Stopwatch
    hud = StopwatchHUD(label=f"Tractus-{args.model_size.upper()} ({args.engine.upper()}) Web Navigation")
    hud.start()

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Executing Autonomous Closed-Loop DOM Navigation...")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    # 4. Run Real Autonomous Navigation
    result = executor.run_autonomous_navigation(max_steps=args.max_steps)

    hud.stop()

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Execution Finished!")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    for item in result["execution_log"]:
        print(
            f"Step {item['step']:02d}: Chose [{item['action_slot']}] {item['chosen_action']:<25} "
            f"| Neural Latency: {item['latency_ms']:6.2f}ms | Conf: {item['confidence']*100:5.1f}%"
        )

    print()
    print(hud.render())
    print()

    print("📊 Real Physical Metrics:")
    print(f"  • Total Neural Actions:    {result['total_actions']}")
    print(f"  • Successful Actions:       {result['successful_actions']}")
    print(f"  • Action Success Rate:      {result['success_rate'] * 100:.1f}%")
    print(f"  • Total Neural Latency:     {result['total_neural_latency_ms']:.1f}ms")
    print(f"  • Avg Latency per Action:   {result['total_neural_latency_ms'] / max(1, result['total_actions']):.1f}ms")
    print(f"  • Form Submission Status:   {'✓ Form Completed & Submitted' if result['web_app_state']['submitted'] else 'In Progress'}")
    print()


if __name__ == "__main__":
    main()
