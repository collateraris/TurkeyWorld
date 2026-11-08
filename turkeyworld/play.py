"""Play the ground-truth prototype, or benchmark it without a window.

This program runs the reference game engine. Its displayed enlargement is
classical image resizing; it does not run trained world-model or neural
super-resolution weights. Use ``python -m turkeyworld.play --help``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .engine import Game

SIM_HZ = 30
SIM_DT = 1.0 / SIM_HZ


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=256,
                        help="Square reference-render resolution (default: 256).")
    parser.add_argument("--display-size", type=int, default=1024,
                        help="Square window/output size (default: 1024).")
    parser.add_argument("--fps", type=int, default=60,
                        help="Window render-rate cap; simulation remains 30 Hz.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="Stop after this duration; 0 means unlimited in a window or 10 simulated seconds headlessly.")
    parser.add_argument("--headless", action="store_true",
                        help="Run a deterministic demo and CPU timing check without pygame.")
    parser.add_argument("--output", type=Path,
                        help="Save the last displayed frame as a PNG.")
    parser.add_argument("--smooth", action="store_true",
                        help="Use bilinear/smooth classical resize instead of crisp nearest-neighbour resize.")
    parser.add_argument("--no-overlays", action="store_true",
                        help="Remove the HUD/crosshair; weapon animation remains in the scene.")
    args = parser.parse_args()
    if not 32 <= args.size <= 2048:
        parser.error("--size must be between 32 and 2048.")
    if not 32 <= args.display_size <= 4096:
        parser.error("--display-size must be between 32 and 4096.")
    if not 1 <= args.fps <= 240:
        parser.error("--fps must be between 1 and 240.")
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error("--seconds must be finite and non-negative.")
    return args


def _enlarge(frame: np.ndarray, size: int, smooth: bool) -> Image.Image:
    mode = Image.Resampling.BILINEAR if smooth else Image.Resampling.NEAREST
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    return image if image.size == (size, size) else image.resize((size, size), mode)


def _save(image: Image.Image, destination: Path | None) -> None:
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, format="PNG")
        print(f"Saved frame: {destination.resolve()}")


def _demo_action(t: float) -> np.ndarray:
    """A reproducible input sequence, not an expert or a learned policy."""
    # Start by shooting the opponents ahead, then explore by moving/turning.
    if t < 2.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    phase = (t - 2.0) % 12.0
    if phase < 4.0:
        return np.array([0.65, 0.12, 0.10, 1.0], dtype=np.float32)
    if phase < 6.0:
        return np.array([0.0, 0.0, 0.65, 1.0], dtype=np.float32)
    if phase < 10.0:
        return np.array([0.5, -0.25, -0.10, 1.0], dtype=np.float32)
    return np.array([0.0, 0.0, -0.7, 0.0], dtype=np.float32)


def _headless(args: argparse.Namespace) -> None:
    game = Game(seed=args.seed)
    seconds = args.seconds or 10.0
    steps = max(1, math.ceil(seconds * SIM_HZ))
    event_counts: dict[str, int] = {}
    step_seconds = render_seconds = resize_seconds = 0.0
    image = None
    started = time.perf_counter()
    for index in range(steps):
        t0 = time.perf_counter()
        events = game.step(_demo_action(index * SIM_DT), dt=SIM_DT)
        t1 = time.perf_counter()
        frame = game.render(size=args.size, overlays=not args.no_overlays)
        t2 = time.perf_counter()
        image = _enlarge(frame, args.display_size, args.smooth)
        t3 = time.perf_counter()
        step_seconds += t1 - t0
        render_seconds += t2 - t1
        resize_seconds += t3 - t2
        # Engine event contents may grow; type names are enough for this report.
        if isinstance(events, list):
            for event in events:
                kind = str(event.get("type", "event")) if isinstance(event, dict) else str(event)
                event_counts[kind] = event_counts.get(kind, 0) + 1
        elif isinstance(events, dict):
            for kind in ("shot", "hit", "kill", "player_hit"):
                event_counts[kind] = event_counts.get(kind, 0) + int(bool(events.get(kind, False)))
    elapsed = time.perf_counter() - started
    state = game.get_state()
    player = state.get("player", {})
    report = {
        "mode": "reference_engine_cpu_classical_upscale",
        "simulation_hz": SIM_HZ,
        "simulated_seconds": round(steps * SIM_DT, 3),
        "rendered_frames": steps,
        "base_resolution": [args.size, args.size],
        "output_resolution": [args.display_size, args.display_size],
        "resize": "bilinear" if args.smooth else "nearest",
        "wall_seconds": round(elapsed, 4),
        "throughput_frames_per_second": round(steps / elapsed, 2),
        "mean_step_ms": round(1000 * step_seconds / steps, 3),
        "mean_render_ms": round(1000 * render_seconds / steps, 3),
        "mean_resize_ms": round(1000 * resize_seconds / steps, 3),
        "player_hp": player.get("hp"),
        "kills": state.get("kills"),
        "events": event_counts,
        "neural_inference_benchmarked": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if image is not None:
        _save(image, args.output)


def _interactive(args: argparse.Namespace) -> None:
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit("Window mode requires pygame: pip install pygame\n"
                         "For a no-window run use --headless.") from exc
    pygame.init()
    screen = pygame.display.set_mode((args.display_size, args.display_size))
    game = Game(seed=args.seed)
    clock = pygame.time.Clock()
    accumulator = 0.0
    running = True
    started = previous = time.perf_counter()
    last_frame = None
    print("W/S: forward/back | A/D: strafe | arrows or Q/E: turn | SPACE/LMB: fire | R: restart | ESC: quit")
    print("Reference engine; 30 Hz simulation; classical image resizing.")
    try:
        while running:
            now = time.perf_counter()
            accumulator += min(now - previous, 0.25)
            previous = now
            if args.seconds and now - started >= args.seconds:
                break
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        game.reset(seed=args.seed)
                        accumulator = 0.0
            if not running:
                break
            keys = pygame.key.get_pressed()
            mouse_fire = pygame.mouse.get_pressed(3)[0]
            action = np.array([
                float(keys[pygame.K_w] or keys[pygame.K_UP]) - float(keys[pygame.K_s] or keys[pygame.K_DOWN]),
                float(keys[pygame.K_d]) - float(keys[pygame.K_a]),
                float(keys[pygame.K_RIGHT] or keys[pygame.K_e]) - float(keys[pygame.K_LEFT] or keys[pygame.K_q]),
                float(keys[pygame.K_SPACE] or mouse_fire),
            ], dtype=np.float32)
            while accumulator >= SIM_DT:
                game.step(action, dt=SIM_DT)
                accumulator -= SIM_DT
            last_frame = game.render(size=args.size, overlays=not args.no_overlays)
            surface = pygame.surfarray.make_surface(last_frame.swapaxes(0, 1))
            if surface.get_size() != (args.display_size, args.display_size):
                scaler = pygame.transform.smoothscale if args.smooth else pygame.transform.scale
                surface = scaler(surface, (args.display_size, args.display_size))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            pygame.display.set_caption(f"TurkeyWorld reference | simulation 30 Hz | display {clock.get_fps():.0f} FPS | classical upscale")
            clock.tick(args.fps)
    finally:
        if last_frame is not None:
            _save(_enlarge(last_frame, args.display_size, args.smooth), args.output)
        pygame.quit()


def main() -> None:
    args = _arguments()
    if args.headless:
        _headless(args)
    else:
        _interactive(args)


if __name__ == "__main__":
    main()
