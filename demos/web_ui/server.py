"""demos/web_ui/server.py

Asynchronous Web Server for Tractus Decision Foundation Models Visual Dashboard.
Built on aiohttp with real-time WebSockets and Apple Silicon Metal GPU acceleration.
Zero external CDN dependencies, self-contained, air-gapped production grade.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

from aiohttp import WSMsgType, web

from .engine_runner import DemoEngineRunner, get_hardware_telemetry

logger = logging.getLogger("TractusWebUI.Server")

STATIC_DIR = Path(__file__).resolve().parent / "static"


class TractusDashboardApp:
    """Dashboard Application orchestrating HTTP REST & WebSocket streams."""

    def __init__(
        self,
        engine_type: str = "mlx",
        model_size: str = "0.8b",
        checkpoint_path: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 8080,
    ):
        self.engine_type = engine_type
        self.model_size = model_size
        self.checkpoint_path = checkpoint_path
        self.host = host
        self.port = port
        self.runner = DemoEngineRunner.get_instance(
            default_engine=engine_type,
            default_model_size=model_size,
            default_checkpoint_path=checkpoint_path,
        )
        self.active_websockets: Set[web.WebSocketResponse] = set()

        # Auto-Run State Machine
        self.autorun_task: Optional[asyncio.Task] = None
        self.autorun_active = False
        self.autorun_paused = False
        self.autorun_speed = 1.0

        self.app = web.Application()
        self._setup_routes()

    def _setup_routes(self):
        self.app.router.add_get("/", self.handle_index)
        self.app.router.add_static("/static", path=str(STATIC_DIR), name="static", follow_symlinks=True)

        # Standalone Demo Pages
        self.app.router.add_get("/demo1", self.handle_demo1_page)
        self.app.router.add_get("/demo2", self.handle_demo2_page)
        self.app.router.add_get("/demo3", self.handle_demo3_page)
        self.app.router.add_get("/demo4", self.handle_demo4_page)

        # REST Endpoints
        self.app.router.add_get("/api/status", self.handle_status)
        self.app.router.add_get("/api/keynote", self.handle_keynote)
        self.app.router.add_get("/api/demo1/scenarios", self.handle_demo1_scenarios)
        self.app.router.add_post("/api/demo1/run", self.handle_demo1_run)
        self.app.router.add_post("/api/demo1/launch_chrome", self.handle_demo1_launch_chrome)
        self.app.router.add_post("/api/demo2/run", self.handle_demo2_run)
        self.app.router.add_post("/api/demo3/run", self.handle_demo3_run)
        self.app.router.add_post("/api/tractus/systemone", self.handle_tractus_systemone)

        # WebSocket Streaming Endpoint
        self.app.router.add_get("/ws/stream", self.handle_websocket)

    async def handle_index(self, request: web.Request) -> web.Response:
        index_file = STATIC_DIR / "index.html"
        if not index_file.exists():
            return web.Response(text="Dashboard frontend not found.", status=404)
        return web.FileResponse(index_file)

    async def handle_demo1_page(self, request: web.Request) -> web.Response:
        demo1_file = STATIC_DIR / "demo1_standalone.html"
        if not demo1_file.exists():
            return web.Response(text="Demo 1 page not found.", status=404)
        return web.FileResponse(demo1_file)

    async def handle_demo2_page(self, request: web.Request) -> web.Response:
        demo2_file = STATIC_DIR / "demo2_standalone.html"
        if not demo2_file.exists():
            return web.Response(text="Demo 2 page not found.", status=404)
        return web.FileResponse(demo2_file)

    async def handle_demo3_page(self, request: web.Request) -> web.Response:
        demo3_file = STATIC_DIR / "demo3_standalone.html"
        if not demo3_file.exists():
            return web.Response(text="Demo 3 page not found.", status=404)
        return web.FileResponse(demo3_file)

    async def handle_demo4_page(self, request: web.Request) -> web.Response:
        demo4_file = STATIC_DIR / "demo4_standalone.html"
        if not demo4_file.exists():
            return web.Response(text="Demo 4 page not found.", status=404)
        return web.FileResponse(demo4_file)

    async def handle_status(self, request: web.Request) -> web.Response:
        telemetry = get_hardware_telemetry()
        telemetry["current_engine"] = self.engine_type
        telemetry["current_model_size"] = self.model_size
        return web.json_response({"status": "ok", "telemetry": telemetry})

    async def handle_keynote(self, request: web.Request) -> web.Response:
        script = self.runner.get_keynote_script()
        return web.json_response({"status": "ok", "script": script})

    async def handle_demo1_scenarios(self, request: web.Request) -> web.Response:
        from demos.demo1_web_automation import list_available_scenarios
        return web.json_response({"status": "ok", "scenarios": list_available_scenarios()})

    async def handle_demo1_run(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            scenario_id = body.get("scenario_id", "google_flights")
        except Exception:
            scenario_id = "google_flights"
        results = []
        async for item in self.runner.run_demo1_stream(self.engine_type, self.model_size, scenario_id=scenario_id, step_delay_s=0.0):
            results.append(item)
        return web.json_response({"status": "ok", "steps": results})

    async def handle_demo1_launch_chrome(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            scenario_id = body.get("scenario_id", "google_flights")
            speed = float(body.get("speed", 1.0))
        except Exception:
            scenario_id = "google_flights"
            speed = 1.0
        asyncio.create_task(self._launch_chrome_async(scenario_id, speed))
        return web.json_response({"status": "launched", "scenario_id": scenario_id})

    async def handle_demo2_run(self, request: web.Request) -> web.Response:
        results = []
        async for item in self.runner.run_demo2_stream(self.engine_type, self.model_size, chunk_delay_s=0.0):
            results.append(item)
        return web.json_response({"status": "ok", "chunks": results})

    async def handle_demo3_run(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            complaint = body.get("complaint", None)
        except Exception:
            complaint = None
        result = await self.runner.run_demo3_analysis(complaint, self.engine_type, self.model_size)
        return web.json_response({"status": "ok", "result": result})

    async def handle_tractus_systemone(self, request: web.Request) -> web.Response:
        """High-throughput System One decision endpoint for JevBall 3D multi-agent decisions."""
        try:
            body = await request.json()
        except Exception as e:
            return web.json_response({"error": f"Invalid JSON body: {e}"}, status=400)

        questions = body.get("questions", {})
        model_tag = f"tractus-{self.model_size}-{self.engine_type}"
        if not questions:
            return web.json_response({
                "model": model_tag,
                "answers": {},
                "usage": {"input_tokens": 0, "output_tokens": 0},
            })

        loop = asyncio.get_running_loop()
        engine = self.runner.get_physical_engine(self.engine_type, self.model_size)

        def _infer_sync():
            answers = {}
            total_tokens = 0
            slot_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]
            for player_id, q in questions.items():
                instructions = q.get("instructions", "")
                criteria = q.get("criteria", {})
                if not criteria:
                    continue

                letter_to_slot = {}
                prompt = (
                    f"Task: Football Tactical Action Selection.\n"
                    f"Agent: {instructions}\n"
                    f"Options:\n"
                )
                for idx, (slot, desc) in enumerate(criteria.items()):
                    let = slot_letters[idx % len(slot_letters)]
                    letter_to_slot[let] = slot
                    prompt += f"{let}: {slot} ({desc})\n"
                prompt += "Decision: ["

                total_tokens += len(prompt) // 4
                res = engine.predict_choice(prompt=prompt, candidate_slots=list(letter_to_slot.keys()))

                chosen_letter = res.chosen_slot
                chosen_slot = letter_to_slot.get(chosen_letter, list(criteria.keys())[0])

                probs_mapped = {}
                for let, p in res.probabilities.items():
                    if let in letter_to_slot:
                        probs_mapped[letter_to_slot[let]] = round(float(p), 4)

                answers[player_id] = {
                    "type": "choice",
                    "choice": chosen_slot,
                    "confidence": round(float(res.confidence), 4),
                    "probabilities": probs_mapped,
                }
            return answers, total_tokens

        t0 = time.perf_counter()
        answers, total_tokens = await loop.run_in_executor(None, _infer_sync)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return web.json_response({
            "model": model_tag,
            "answers": answers,
            "usage": {
                "input_tokens": total_tokens,
                "output_tokens": len(answers) * 3,
            },
            "elapsed_ms": round(elapsed_ms, 2),
        })

    async def broadcast_ws(self, msg: Dict[str, Any]):
        """Broadcasts a JSON message to all connected dashboard clients."""
        payload = json.dumps(msg)
        dead = []
        for ws in self.active_websockets:
            try:
                await ws.send_str(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active_websockets.discard(ws)

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20.0)
        await ws.prepare(request)
        self.active_websockets.add(ws)
        logger.info(f"WebSocket client connected. Total clients: {len(self.active_websockets)}")

        # Send initial telemetry handshake
        initial_telemetry = get_hardware_telemetry()
        initial_telemetry["current_engine"] = self.engine_type
        initial_telemetry["current_model_size"] = self.model_size
        await ws.send_str(json.dumps({"type": "telemetry", "data": initial_telemetry}))

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        cmd = json.loads(msg.data)
                        action = cmd.get("action", "")
                        await self._dispatch_ws_command(action, cmd, ws)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON received: {msg.data}")
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
        finally:
            self.active_websockets.discard(ws)
            logger.info(f"WebSocket client disconnected. Remaining: {len(self.active_websockets)}")
        return ws

    async def _dispatch_ws_command(self, action: str, cmd: Dict[str, Any], ws: web.WebSocketResponse):
        """Processes control actions from client."""
        if action == "ping":
            await ws.send_str(json.dumps({"type": "pong", "time": time.time()}))

        elif action == "set_engine":
            new_eng = cmd.get("engine", self.engine_type)
            new_model = cmd.get("model_size", self.model_size)
            self.engine_type = new_eng
            self.model_size = new_model
            logger.info(f"Engine switched to: {self.engine_type} | {self.model_size}")
            await self.broadcast_ws({
                "type": "engine_switched",
                "engine": self.engine_type,
                "model_size": self.model_size,
            })

        elif action == "run_demo1":
            speed = float(cmd.get("speed", 1.0))
            scenario_id = str(cmd.get("scenario_id", "google_flights"))
            delay = max(0.01, 0.08 / speed)
            async for step_event in self.runner.run_demo1_stream(self.engine_type, self.model_size, scenario_id=scenario_id, step_delay_s=delay):
                await self.broadcast_ws(step_event)

        elif action == "launch_playwright_chrome":
            scenario_id = str(cmd.get("scenario_id", "google_flights"))
            speed = float(cmd.get("speed", 1.0))
            asyncio.create_task(self._launch_chrome_async(scenario_id, speed))

        elif action == "run_demo2":
            speed = float(cmd.get("speed", 1.0))
            delay = max(0.02, 0.22 / speed)
            async for chunk_event in self.runner.run_demo2_stream(self.engine_type, self.model_size, chunk_delay_s=delay):
                await self.broadcast_ws(chunk_event)

        elif action == "run_demo3":
            complaint = cmd.get("complaint", None)
            res = await self.runner.run_demo3_analysis(complaint, self.engine_type, self.model_size)
            await self.broadcast_ws({"type": "demo3_result", "data": res})

        elif action == "start_autorun":
            speed = float(cmd.get("speed", 1.0))
            self.autorun_speed = speed
            if self.autorun_task and not self.autorun_task.done():
                self.autorun_task.cancel()
            self.autorun_task = asyncio.create_task(self._run_keynote_presentation())

        elif action == "pause_autorun":
            self.autorun_paused = True
            await self.broadcast_ws({"type": "autorun_paused"})

        elif action == "resume_autorun":
            self.autorun_paused = False
            await self.broadcast_ws({"type": "autorun_resumed"})

        elif action == "stop_autorun":
            if self.autorun_task and not self.autorun_task.done():
                self.autorun_task.cancel()
            self.autorun_active = False
            self.autorun_paused = False
            await self.broadcast_ws({"type": "autorun_stopped"})

    async def _launch_chrome_async(self, scenario_id: str, speed: float = 1.0):
        """Asynchronously executes native Chrome desktop automation via Playwright."""
        await self.broadcast_ws({
            "type": "native_chrome_status",
            "data": {
                "status": "launching",
                "scenario_id": scenario_id,
                "message": f"Launching native Google Chrome on macOS desktop [{scenario_id}]...",
            }
        })
        from demos.demo1_web_automation.playwright_runner import TractusNativePlaywrightRunner

        def _worker():
            runner = TractusNativePlaywrightRunner(
                scenario=scenario_id,
                headless=False,
                engine_type=self.engine_type,
                model_size=self.model_size,
                speed=speed,
                port=self.port,
                checkpoint_path=self.checkpoint_path,
            )
            return runner.run(keep_open_seconds=3.0)

        loop = asyncio.get_running_loop()
        try:
            summary = await loop.run_in_executor(None, _worker)
            await self.broadcast_ws({
                "type": "native_chrome_status",
                "data": {
                    "status": "completed",
                    "scenario_id": scenario_id,
                    "summary": summary,
                    "message": "Native Google Chrome physical desktop automation completed successfully!",
                }
            })
        except Exception as e:
            logger.error("Failed to execute native Chrome: %s", e)
            await self.broadcast_ws({
                "type": "native_chrome_status",
                "data": {
                    "status": "error",
                    "scenario_id": scenario_id,
                    "error": str(e),
                    "message": f"Native Chrome launch exception: {e}",
                }
            })

    async def _run_keynote_presentation(self):
        """Orchestrates the entire keynote presentation timeline automatically."""
        self.autorun_active = True
        self.autorun_paused = False
        script = self.runner.get_keynote_script()

        logger.info("🎬 Commencing Auto-Run Keynote Presentation...")
        await self.broadcast_ws({"type": "autorun_started", "script": script})

        try:
            for idx, stage in enumerate(script):
                while self.autorun_paused:
                    await asyncio.sleep(0.2)

                await self.broadcast_ws({
                    "type": "keynote_stage",
                    "stage_index": idx,
                    "stage": stage,
                    "target_tab": stage["target_tab"],
                })

                stage_id = stage["stage_id"]
                speed = self.autorun_speed

                if stage_id == "intro":
                    await asyncio.sleep(stage["duration_s"] / speed)

                elif stage_id == "demo1":
                    delay = max(0.02, 0.08 / speed)
                    async for step_event in self.runner.run_demo1_stream(self.engine_type, self.model_size, step_delay_s=delay):
                        while self.autorun_paused:
                            await asyncio.sleep(0.2)
                        await self.broadcast_ws(step_event)
                    await asyncio.sleep(1.0 / speed)

                elif stage_id == "demo2":
                    delay = max(0.04, 0.22 / speed)
                    async for chunk_event in self.runner.run_demo2_stream(self.engine_type, self.model_size, chunk_delay_s=delay):
                        while self.autorun_paused:
                            await asyncio.sleep(0.2)
                        await self.broadcast_ws(chunk_event)
                    await asyncio.sleep(1.0 / speed)

                elif stage_id == "demo3":
                    res = await self.runner.run_demo3_analysis(None, self.engine_type, self.model_size)
                    await self.broadcast_ws({"type": "demo3_result", "data": res})
                    await asyncio.sleep(stage["duration_s"] / speed)

                elif stage_id == "summary":
                    await asyncio.sleep(stage["duration_s"] / speed)

            await self.broadcast_ws({"type": "autorun_finished"})
            logger.info("✔ Auto-Run Keynote Presentation completed successfully.")

        except asyncio.CancelledError:
            logger.info("Auto-Run Keynote Presentation cancelled.")
        finally:
            self.autorun_active = False
            self.autorun_paused = False


async def create_app(engine: str = "mlx", model_size: str = "0.8b") -> web.Application:
    dashboard = TractusDashboardApp(engine_type=engine, model_size=model_size)
    return dashboard.app
