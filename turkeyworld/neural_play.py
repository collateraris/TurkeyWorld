"""Play a loaded neural world checkpoint without ground-truth corrections.

The reference engine supplies a map and initial single-enemy20-value state once.
V-JEPA models additionally encode one initial source RGB frame before rollout.
Every subsequent transition and base image comes from the loaded world model.
No trained checkpoints are bundled. A successful run proves executable shapes,
not correct combat, stable gameplay or RTX 4090 performance.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .engine import Game
from .models import (ENTITY_DIM, ENTITY_SLOTS, GLOBAL_DIM, STATE_DIM,
                     load_sr_checkpoint, load_world_checkpoint, parameter_counts)

SIM_HZ = 30
SIM_DT = 1.0 / SIM_HZ
EVENT_NAMES = ("shot", "hit", "kill", "player_hit")


def _arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world", type=Path, required=True, help="World checkpoint produced by turkeyworld.train.")
    p.add_argument("--renderer-checkpoint", type=Path,
                   help="Optional RF renderer bundle; preserves teacher dynamics, validates source conditions and approved compact assets.")
    p.add_argument("--base-checkpoint", type=Path,
                   help="Optional validated compact Base bundle; renderer bundles may also embed it automatically.")
    p.add_argument("--allow-unqualified-codec", action='store_true',
                   help="Explicit experimental preview of an unqualified compact codec; preserves failed validation metadata.")
    p.add_argument("--sr", type=Path, help="Optional trained x4 SR checkpoint; otherwise use classical resize.")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--fp16", action="store_true", help="Half precision; supported here only on CUDA.")
    p.add_argument("--headless", action="store_true", help="Run a reproducible input sequence without a window or real-time pacing.")
    p.add_argument("--steps", type=int, default=0, help="Stop after N simulated 1/30-second steps; headless default is 300, window default unlimited.")
    p.add_argument("--seconds", type=float, default=0, help="Optional window wall-clock duration limit.")
    p.add_argument("--fps", type=int, choices=(30, 60), default=60, help="Display cap. At 60 FPS the latest 30 Hz image is repeated, without interpolation.")
    p.add_argument("--display-size", type=int, default=None,
                   help="Window/output size: defaults to native*16 for two-stage playback, otherwise1024.")
    p.add_argument("--two-stage-upscale", action="store_true",
                   help="Native32/64 RGB -> trained neural SR x4 -> bilinear x4; requires --sr.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=3, help="Untimed warmup steps; their predicted state is discarded.")
    p.add_argument("--threads", type=int, default=4, help="CPU PyTorch thread count.")
    p.add_argument("--max-catchup-steps", type=int, default=None,
                   help="Maximum neural updates before polling input; defaults to1 for two-stage playback, otherwise4.")
    p.add_argument("--smooth", action="store_true", help="Bilinear rather than nearest classical resize.")
    p.add_argument("--no-hud", action="store_true", help="Hide interactive HUD and crosshair; headless frames are always clean.")
    p.add_argument("--output", type=Path, help="Save the last neural frame as PNG.")
    p.add_argument("--stats", type=Path, help="Also write the JSON report to this path.")
    a = p.parse_args(argv)
    if a.max_catchup_steps is None:
        a.max_catchup_steps = 1 if a.two_stage_upscale else 4
    if a.display_size is None and not a.two_stage_upscale:
        a.display_size = 1024
    if a.steps < 0 or a.warmup < 0:
        p.error("--steps and --warmup must be non-negative")
    if not math.isfinite(a.seconds) or a.seconds < 0:
        p.error("--seconds must be finite and non-negative")
    if a.display_size is not None and not 32 <= a.display_size <= 4096:
        p.error("--display-size must be between 32 and 4096")
    if a.two_stage_upscale and a.sr is None:
        p.error("--two-stage-upscale requires a trained --sr checkpoint")
    if a.threads < 1:
        p.error("--threads must be positive")
    if a.max_catchup_steps < 1:
        p.error("--max-catchup-steps must be positive")
    if a.fp16 and a.device != "cuda":
        p.error("--fp16 requires --device cuda")
    if a.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA was requested but is unavailable; choose --device cpu")
    if not a.world.is_file():
        p.error(f"world checkpoint not found: {a.world}; train a checkpoint first")
    if a.renderer_checkpoint is not None and not a.renderer_checkpoint.is_file():
        p.error(f"renderer bundle not found: {a.renderer_checkpoint}")
    if a.allow_unqualified_codec and a.renderer_checkpoint is None:
        p.error('--allow-unqualified-codec requires --renderer-checkpoint')
    if a.base_checkpoint is not None and not a.base_checkpoint.is_file():
        p.error(f"Base bundle not found: {a.base_checkpoint}")
    if a.sr is not None and not a.sr.is_file():
        p.error(f"SR checkpoint not found: {a.sr}; omit --sr for classical resize")
    return a


def _demo_action(t):
    """Fixed test inputs, not an engine-dependent or trained policy."""
    if t < 2:
        return [0.0, 0.0, 0.0, 1.0]
    phase = (t - 2) % 12
    if phase < 4:
        return [0.65, 0.12, 0.10, 1.0]
    if phase < 6:
        return [0.0, 0.0, 0.65, 1.0]
    if phase < 10:
        return [0.5, -0.25, -0.10, 1.0]
    return [0.0, 0.0, -0.7, 0.0]


class NeuralRuntime:
    """Own learned state; the engine object is discarded after initialization."""
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        torch.set_num_threads(args.threads)
        self.model, self.world_metadata = load_world_checkpoint(args.world, self.device)
        if getattr(args, 'base_checkpoint', None) is not None:
            from .base_student import load_base_student
            load_base_student(self.model, args.base_checkpoint)
        self.renderer_metadata = None
        if getattr(args, 'renderer_checkpoint', None) is not None:
            from .renderer_student import load_renderer_student
            self.renderer_metadata = load_renderer_student(self.model, args.renderer_checkpoint,
                allow_unqualified_codec=getattr(args, 'allow_unqualified_codec', False))
        if getattr(args, "two_stage_upscale", False):
            native = self.model.config.image_size
            if args.sr is None or native not in (32, 64):
                raise ValueError("Two-stage playback requires native32/64 and a trained SR checkpoint")
            if args.display_size is None:
                args.display_size = native*16
            if args.display_size != native*16:
                raise ValueError("Two-stage display size must equal native model size times16")
        self.simulation_hz = float(self.model.config.simulation_hz)
        self.simulation_dt = 1.0 / self.simulation_hz
        self.sr, self.sr_metadata = (None, None)
        if args.sr is not None:
            self.sr, self.sr_metadata = load_sr_checkpoint(args.sr, self.device)
        if getattr(args, "two_stage_upscale", False):
            trained = self.sr_metadata.get("training_steps", 0)
            if (not isinstance(trained, (int, float)) or isinstance(trained, bool)
                    or not math.isfinite(trained) or trained < 1):
                raise ValueError("Two-stage playback requires an SR checkpoint with positive training_steps")
            for key, expected in (("input_size", self.model.config.image_size),
                                  ("output_size", self.model.config.image_size*4)):
                if key in self.sr_metadata and self.sr_metadata[key] != expected:
                    raise ValueError("SR checkpoint source/output resolution differs from the native world model")
        self.dtype = torch.float16 if args.fp16 else torch.float32
        self.state_dtype = (torch.float32 if self.model.config.transition_mode == "event_v1"
                            else self.dtype)
        if args.fp16:
            self.model.half()
            if self.sr is not None:
                self.sr.half()
        # This is the only interaction with the reference game. No engine step,
        # render, hit test or state correction is called during neural rollout.
        initialization_game = Game(seed=args.seed, endless=self.model.config.endless,
            enemies_enabled=not getattr(self.model.config, 'navigation_only', False))
        initial_map = initialization_game.map.copy()
        initial_state = initialization_game.state_vector().copy()
        self.world_architecture = getattr(self.model.config, "architecture", "legacy_cnn")
        initial_rgb = (initialization_game.render(size=self.model.config.image_size, overlays=False)
                       if self.world_architecture == "vjepa2_ac" else None)
        self.engine_renders_for_initialization = int(initial_rgb is not None)
        del initialization_game
        if initial_state.shape != (STATE_DIM,):
            raise ValueError(f"initial state must have {STATE_DIM} values")
        self.initial_state = torch.as_tensor(initial_state, device=self.device, dtype=self.state_dtype)[None]
        grid = torch.as_tensor(initial_map, device=self.device, dtype=torch.long)[None]
        with torch.inference_mode():
            self.map_features = self.model.encode_map(grid)
            if initial_rgb is not None:
                image = torch.as_tensor(initial_rgb, device=self.device).permute(2, 0, 1)[None]
                image = image.to(dtype=self.dtype).div(255)
                initial_memory = self.model.initialize_from_image(image, self.initial_state, self.map_features)
            else:
                initial_memory = self.model.initial_memory(1, self.device, self.dtype)
            if (not torch.is_tensor(initial_memory) or initial_memory.ndim < 2
                    or initial_memory.shape[0] != 1 or not bool(torch.isfinite(initial_memory).all())):
                raise ValueError("Initial neural memory must be a finite tensor with batch dimension1")
            self.initial_memory = initial_memory.detach().clone()
        self.steps = 0
        self.samples = defaultdict(list)
        self.event_counts = dict.fromkeys(EVENT_NAMES, 0)
        self.last_frame = None
        self.upscale_stages = []
        self.reset_from_cached_initial_state()

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def reset_from_cached_initial_state(self):
        """Used once after warmup; does not query the original engine."""
        self.state = self.initial_state.clone()
        self.memory = self.initial_memory.clone()

    def _resize(self, rgb):
        self.upscale_stages = []
        if getattr(self.args, "two_stage_upscale", False):
            native = self.model.config.image_size
            if native not in (32, 64) or rgb.shape[-2:] != (native, native) or self.sr is None:
                raise ValueError("Two-stage playback requires native32/64 RGB and a trained neural x4 SR model")
            rgb = self.sr(rgb)
            middle, final = native*4, native*16
            if rgb.shape[-2:] != (middle, middle):
                raise ValueError("Neural SR must produce exactly x4 native resolution")
            self.upscale_stages.append({"method": "neural_x4", "input_resolution": [native, native],
                                        "output_resolution": [middle, middle], "scale": 4})
            rgb = F.interpolate(rgb, size=(final, final), mode="bilinear", align_corners=False)
            self.upscale_stages.append({"method": "bilinear", "input_resolution": [middle, middle],
                                        "output_resolution": [final, final], "scale": 4,
                                        "align_corners": False})
            return rgb
        if self.sr is not None:
            native = list(rgb.shape[-2:])
            rgb = self.sr(rgb)
            self.upscale_stages.append({"method": "neural_x4", "input_resolution": native,
                                        "output_resolution": list(rgb.shape[-2:]), "scale": 4})
        if rgb.shape[-2:] != (self.args.display_size, self.args.display_size):
            mode = "bilinear" if self.args.smooth else "nearest"
            options = {"align_corners": False} if mode == "bilinear" else {}
            native = list(rgb.shape[-2:])
            rgb = F.interpolate(rgb, size=(self.args.display_size, self.args.display_size), mode=mode, **options)
            self.upscale_stages.append({"method": mode, "input_resolution": native,
                                        "output_resolution": list(rgb.shape[-2:]), **options})
        return rgb

    @torch.inference_mode()
    def step(self, action, record=True):
        self.synchronize()
        started = time.perf_counter()
        action_tensor = torch.as_tensor(action, device=self.device, dtype=self.state_dtype).reshape(1, 4)
        result = self.model.step(self.state, action_tensor, memory=self.memory,
                                 map_features=self.map_features)
        self.synchronize()
        model_end = time.perf_counter()
        output = self._resize(result["rgb"])
        self.synchronize()
        upscale_end = time.perf_counter()
        if not bool(torch.isfinite(result["state"]).all()) or not bool(torch.isfinite(output).all()):
            raise FloatingPointError(f"non-finite neural prediction at step {self.steps + 1}")
        self.state, self.memory = result["state"], result["memory"]
        # Include actual device->CPU transfer in separately reported timing.
        frame = output[0].clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
        if "events" in result:
            predicted_events = result["events"][0].bool().cpu().tolist()
        else:
            predicted_events = (result["event_logits"][0].sigmoid() >= 0.5).cpu().tolist()
        finished = time.perf_counter()
        self.last_frame = frame
        if record:
            self.steps += 1
            self.samples["model_step_ms"].append(1000 * (model_end - started))
            self.samples["upscale_ms"].append(1000 * (upscale_end - model_end))
            self.samples["readback_and_checks_ms"].append(1000 * (finished - upscale_end))
            self.samples["pipeline_ms"].append(1000 * (finished - started))
            for name, value in zip(EVENT_NAMES, predicted_events):
                self.event_counts[name] += int(value)
        return frame

    def warmup(self):
        for _ in range(self.args.warmup):
            self.step([0, 0, 0, 0], record=False)
        self.reset_from_cached_initial_state()
        self.last_frame = None

    def report(self, wall_seconds, displayed_frames=0, max_backlog=0):
        vector = self.state[0].detach().float().cpu().numpy()
        entities = vector[GLOBAL_DIM:-2].reshape(ENTITY_SLOTS, ENTITY_DIM)
        timing = {}
        for name, values in self.samples.items():
            timing[name] = {"mean": round(float(np.mean(values)), 4),
                            "p50": round(float(np.percentile(values, 50)), 4),
                            "p95": round(float(np.percentile(values, 95)), 4)}
        resize = "bilinear" if self.args.smooth else "nearest"
        sr_kind = "neural_x4" if self.sr is not None else f"classical_{resize}"
        if self.sr is not None and self.model.config.image_size * 4 != self.args.display_size:
            sr_kind += f"_then_classical_{resize}"
        if getattr(self.args, "two_stage_upscale", False):
            sr_kind = "neural_x4_then_bilinear_x4"
        training_steps = int(self.world_metadata.get("training_steps", 0))
        return {
            "mode": "standalone_neural_world_rollout",
            "device": str(self.device),
            "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU",
            "precision": "fp16" if self.args.fp16 else "fp32",
            "persistent_state_precision": str(self.state.dtype),
            "world_checkpoint": str(self.args.world.resolve()),
            "renderer_checkpoint": str(self.args.renderer_checkpoint.resolve()) if getattr(self.args, 'renderer_checkpoint', None) else None,
            "renderer_training_steps": self.renderer_metadata.get('training_steps') if self.renderer_metadata else None,
            "renderer_solver": 'euler' if getattr(self.model, 'flow_renderer', None) is not None and self.model.flow_renderer.kind == 'student' else 'heun' if getattr(self.model, 'flow_renderer', None) is not None else None,
            "renderer_solver_steps": getattr(getattr(self.model, 'flow_renderer', None), 'steps', None),
            "base_student_parameters": getattr(self.model, 'base_student_metadata', {}).get('head_parameters'),
            "base_student_training_steps": getattr(self.model, 'base_student_metadata', {}).get('training_steps'),
            "experimental_codec_preview": bool(self.renderer_metadata and self.renderer_metadata.get('experimental_preview') is True),
            "codec_quality_qualified": self.renderer_metadata.get('codec_quality_qualified') if self.renderer_metadata else None,
            "world_architecture": self.world_architecture,
            "world_training_steps": training_steps,
            "world_quality_status": self.world_metadata.get("quality_status", "Unverified checkpoint"),
            "untrained_world_checkpoint": training_steps == 0,
            "sr_checkpoint": str(self.args.sr.resolve()) if self.args.sr else None,
            "sr_training_steps": None if self.sr_metadata is None else self.sr_metadata.get("training_steps", 0),
            "upscale": sr_kind,
            "upscale_stages": getattr(self, "upscale_stages", []),
            "base_resolution": [self.model.config.image_size] * 2,
            "output_resolution": [self.args.display_size] * 2,
            "simulation_hz": self.simulation_hz,
            "transition_mode": self.model.config.transition_mode,
            "simulated_steps": self.steps,
            "simulated_seconds": round(self.steps * self.simulation_dt, 4),
            "wall_seconds_excluding_load_and_warmup": round(wall_seconds, 4),
            "neural_updates_per_wall_second": round(self.steps / wall_seconds, 3) if wall_seconds else None,
            "headless_unpaced": self.args.headless,
            "display_fps_cap": None if self.args.headless else self.args.fps,
            "displayed_frames": displayed_frames,
            "display_behavior": "latest generated frame repeated; no interpolation",
            "max_catchup_steps_per_input_poll": getattr(self.args, "max_catchup_steps", 4),
            "max_simulation_backlog_seconds": round(max_backlog, 4),
            "timing_ms": timing,
            "warmup_steps_excluded": self.args.warmup,
            "engine_initializations": 1,
            "engine_renders_for_initialization": self.engine_renders_for_initialization,
            "engine_steps_during_rollout": 0,
            "engine_renders_during_rollout": 0,
            "ground_truth_corrections": 0,
            "state_dimension": len(vector),
            "predicted_player_hp": round(float(vector[4] * 100), 4),
            "predicted_ammo": round(float(vector[5] * 99), 4),
            "predicted_alive_slots_at_threshold_0_5": int((entities[:, 5] >= 0.5).sum()),
            "predicted_kills_continuous": round(float(vector[-2] * ENTITY_SLOTS), 4),
            "predicted_events_at_threshold_0_5": self.event_counts,
            "final_predicted_state": vector.tolist(),
            "parameter_counts": parameter_counts(self.model, self.sr),
            "gameplay_correctness_evaluated": False,
            "note": "Predicted state and events are model outputs, not verified game facts. Timing applies only to the device reported above; no RTX 4090 result is implied.",
        }


def _headless(runtime, args):
    steps = args.steps or 300
    started = time.perf_counter()
    for index in range(steps):
        runtime.step(_demo_action(index * runtime.simulation_dt))
    return runtime.report(time.perf_counter() - started)


def _draw_hud(screen, vector, font, *, endless):
    """Display predicted counters without changing neural state or RGB targets."""
    import pygame
    width, height = screen.get_size()
    hp = int(round(float(vector[4])*100))
    ammo = "INF" if endless else str(int(round(float(vector[5])*99)))
    kills = int(round(float(vector[-2])*ENTITY_SLOTS))
    living = int((vector[GLOBAL_DIM:-2].reshape(ENTITY_SLOTS, ENTITY_DIM)[:, 5] >= .5).sum())
    text = font.render(f"HP {hp}    AMMO {ammo}    ALIVE {living}    KILLS {kills}", True, (255, 239, 190))
    pygame.draw.rect(screen, (15, 30, 33), (0, 0, width, font.get_height()+12))
    screen.blit(text, (8, 6))
    cx, cy = width//2, height//2
    arm, gap = max(4, min(width, height)//128), 2
    color = (250, 250, 220)
    for start, end in [((cx-arm, cy), (cx-gap, cy)), ((cx+gap, cy), (cx+arm, cy)),
                       ((cx, cy-arm), (cx, cy-gap)), ((cx, cy+gap), (cx, cy+arm))]:
        pygame.draw.line(screen, color, start, end, max(1, width//512))
    if hp <= 0:
        message = font.render("GAME OVER | relaunch to restart", True, (255, 224, 130))
        box = message.get_rect(center=(cx, cy+font.get_height()*2))
        pygame.draw.rect(screen, (25, 30, 35), box.inflate(16, 12))
        screen.blit(message, box)


def _interactive(runtime, args):
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit("Window mode requires pygame; install requirements.txt or use --headless") from exc
    pygame.init()
    screen = pygame.display.set_mode((args.display_size, args.display_size))
    font = pygame.font.Font(None, max(16, min(32, args.display_size//30)))
    show_hud = not getattr(args, "no_hud", False)
    kind = runtime.model.config.kind.upper()
    renderer = runtime.model.config.renderer_mode.replace("_", " ").upper()
    gameplay = "ENDLESS" if runtime.model.config.endless else "CLASSIC"
    clock = pygame.time.Clock()
    screen.fill((0, 0, 0))
    pygame.display.flip()
    print("W/S: move | A/D: strafe | arrows or Q/E: turn | SPACE/LMB: fire | ESC: quit")
    print(f"Standalone neural rollout: {runtime.simulation_hz:g} Hz simulation; latest frame repeated for display; no resets or engine corrections.")
    print("To restart after death, close the window and launch the same command again.")
    accumulator = 0.0
    displayed = 0
    max_backlog = 0.0
    running = True
    started = previous = time.perf_counter()
    cached_surface = None
    try:
        while running:
            now = time.perf_counter()
            accumulator += now - previous
            previous = now
            if args.seconds and now - started >= args.seconds:
                break
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
            if not running:
                break
            keys = pygame.key.get_pressed()
            action = [
                float(keys[pygame.K_w] or keys[pygame.K_UP]) - float(keys[pygame.K_s] or keys[pygame.K_DOWN]),
                float(keys[pygame.K_d]) - float(keys[pygame.K_a]),
                float(keys[pygame.K_RIGHT] or keys[pygame.K_e]) - float(keys[pygame.K_LEFT] or keys[pygame.K_q]),
                float(keys[pygame.K_SPACE] or pygame.mouse.get_pressed(3)[0]),
            ]
            # Bound work per display iteration without discarding elapsed time.
            # If inference is slow, backlog is retained and reported explicitly.
            caught_up = 0
            while accumulator >= runtime.simulation_dt and caught_up < getattr(args, "max_catchup_steps", 4):
                frame = runtime.step(action)
                cached_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                accumulator -= runtime.simulation_dt
                caught_up += 1
                if args.steps and runtime.steps >= args.steps:
                    running = False
                    break
            max_backlog = max(max_backlog, accumulator)
            if cached_surface is not None:
                screen.blit(cached_surface, (0, 0))
                if show_hud:
                    vector = runtime.state[0].detach().float().cpu().numpy()
                    _draw_hud(screen, vector, font, endless=runtime.model.config.endless)
                pygame.display.flip()
                displayed += 1
            pygame.display.set_caption(
                f"TurkeyWorld NEURAL | {kind} {renderer} | {gameplay} | updates {runtime.steps} | {runtime.simulation_hz:g} Hz target | display {clock.get_fps():.0f} FPS")
            clock.tick(args.fps)
    finally:
        pygame.quit()
    return runtime.report(time.perf_counter() - started, displayed, max_backlog)


def main():
    args = _arguments()
    runtime = NeuralRuntime(args)
    training_steps = int(runtime.world_metadata.get("training_steps", 0))
    print(f"Loaded {runtime.model.config.kind} checkpoint: {training_steps} training steps; gameplay accuracy is unverified.")
    if training_steps == 0:
        print("UNTRAINED WEIGHTS: this run is a shape/execution smoke test only.")
    print("Upscaling:", "loaded neural x4 checkpoint" if runtime.sr is not None else "classical resize; no SR checkpoint loaded")
    runtime.warmup()
    report = _headless(runtime, args) if args.headless else _interactive(runtime, args)
    if args.output is not None and runtime.last_frame is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(runtime.last_frame).save(args.output, format="PNG")
        report["saved_frame"] = str(args.output.resolve())
    if args.stats is not None:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
