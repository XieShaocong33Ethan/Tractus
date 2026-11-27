#!/usr/bin/env python3
"""demos/run_web_ui.py

Tractus Decision Foundation Models - Product Keynote Visual Dashboard & Screen-Record Center.
Launches the asynchronous Web visualization interface on Apple Silicon Metal GPU.
Zero external CDN dependencies, self-contained, air-gapped production grade.

Usage:
    python demos/run_web_ui.py
    python demos/run_web_ui.py --port 8080 --engine mlx --model-size 0.8b
"""

import argparse
import asyncio
import os
import socket
import sys
import webbrowser
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEMOS_DIR = Path(__file__).resolve().parent
if str(DEMOS_DIR) not in sys.path:
    sys.path.insert(0, str(DEMOS_DIR))

from aiohttp import web
from demos.web_ui.engine_runner import get_hardware_telemetry
from demos.web_ui.server import TractusDashboardApp

BANNER = r"""
████████╗██████╗  █████╗  ██████╗████████╗██╗   ██╗███████╗
╚══██╔══╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██║   ██║██╔════╝
   ██║   ██████╔╝███████║██║        ██║   ██║   ██║███████╗
   ██║   ██╔══██╗██╔══██║██║        ██║   ██║   ██║╚════██║
   ██║   ██║  ██║██║  ██║╚██████╗   ██║   ╚██████╔╝███████║
   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚══════╝
    KEYNOTE PRODUCT LAUNCH DASHBOARD & SCREEN-RECORD CENTER
"""


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def find_available_port(start_port: int, max_attempts: int = 10, host: str = "127.0.0.1") -> int:
    port = start_port
    while is_port_in_use(port, host) and port < start_port + max_attempts:
        port += 1
    return port


def parse_args():
    parser = argparse.ArgumentParser(description="Tractus Decision Models - High-Performance Visual Telemetry Cockpit")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080, auto-fallback)")
    parser.add_argument("--engine", type=str, default="mlx", choices=["mlx", "pytorch"], help="Inference engine backend (default: mlx)")
    parser.add_argument("--model-size", type=str, default="0.8b", choices=["0.8b", "2b", "4b", "9b"], help="Model size (default: 0.8b)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom path to model checkpoint weights directory")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically launch web browser")
    return parser.parse_args()


def print_launch_banner(host: str, port: int, engine: str, model_size: str, checkpoint: Optional[str], telemetry: dict):
    print("\033[1;36m" + BANNER + "\033[0m")
    print("\033[1;37m" + "─" * 78 + "\033[0m")
    print(f"  \033[1;32m● Hardware Node\033[0m       : {telemetry.get('chip', 'Apple Silicon')} ({telemetry.get('total_memory_gb', 24)}GB Unified)")
    print(f"  \033[1;32m● GPU Acceleration\033[0m    : Native Metal Shaders (MLX C++ & PyTorch MPS)")
    print(f"  \033[1;32m● Active Inference\033[0m    : \033[1;33m{engine.upper()}\033[0m backend | \033[1;33mTractus-{model_size.upper()}\033[0m")
    if checkpoint:
        print(f"  \033[1;32m● Custom Checkpoint\033[0m   : \033[1;34m{checkpoint}\033[0m")
    print(f"  \033[1;32m● Physical Standard\033[0m   : \033[1;35m100% Real Neural Inference (Zero Synthetic Mock)\033[0m")
    print("─" * 78)
    print(f"  \033[1;32m✔ Local Cockpit URL\033[0m   : \033[1;36mhttp://{host}:{port}\033[0m")
    print(f"  \033[1;32m✔ Standalone Demo 1\033[0m   : \033[1;36mhttp://{host}:{port}/demo1\033[0m (Web Agent)")
    print(f"  \033[1;32m✔ Standalone Demo 2\033[0m   : \033[1;36mhttp://{host}:{port}/demo2\033[0m (Meeting Sentinel)")
    print(f"  \033[1;32m✔ Standalone Demo 3\033[0m   : \033[1;36mhttp://{host}:{port}/demo3\033[0m (Service Switchboard)")
    print(f"  \033[1;32m✔ Standalone Demo 4\033[0m   : \033[1;36mhttp://{host}:{port}/demo4\033[0m (Spatial Dynamics)")
    print("\033[1;37m" + "─" * 78 + "\033[0m")
    print("  \033[0;37mPress Ctrl+C to stop the dashboard server.\033[0m\n")


def main():
    args = parse_args()
    port = find_available_port(args.port, host=args.host)
    if port != args.port:
        print(f"\033[1;33mPort {args.port} was occupied, automatically shifted to port {port}.\033[0m")

    # Verify authentic audio assets are available
    audio_manifest_file = PROJECT_ROOT / "demos" / "web_ui" / "static" / "audio" / "manifest.json"
    if not audio_manifest_file.exists():
        print(f"Warning: Audio manifest not found at {audio_manifest_file}")

    telemetry = get_hardware_telemetry()
    print_launch_banner(args.host, port, args.engine, args.model_size, args.checkpoint, telemetry)

    dashboard_app = TractusDashboardApp(
        engine_type=args.engine,
        model_size=args.model_size,
        checkpoint_path=args.checkpoint,
        host=args.host,
        port=port,
    )

    if not args.no_browser:
        target_url = f"http://{args.host}:{port}"
        try:
            # Short async timer to open browser once server loop starts
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.call_later(0.5, lambda: webbrowser.open(target_url))
        except Exception:
            pass

    web.run_app(dashboard_app.app, host=args.host, port=port, print=lambda *args: None)


if __name__ == "__main__":
    main()
