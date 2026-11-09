"""Measure free-rollout integrity, active coverage, and optional sampled RGB.

The reference engine prepares a fixed action tape before neural execution. Its
states are measurement targets, never corrections to the model or GRU memory.
A completed rollout alone does not establish active gameplay for its duration.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from .engine import Game
from .runtime_checks import EVENT_NAMES, RolloutMonitor, entities

SCENARIOS = ("idle", "wall", "turn", "explore", "combat", "mixed")
WAYPOINTS = np.array([[14.25, 1.75], [14.25, 14.25],
                      [1.75, 14.25], [1.75, 1.75]], dtype=np.float64)


def scripted_action(state, scenario, step, dt, waypoint):
    """A source-only controller used to record a fixed paired tape."""
    if scenario == "idle":
        return np.zeros(4, dtype=np.float32), waypoint
    if scenario == "wall":
        return np.array([1., 0., 0., 0.], dtype=np.float32), waypoint
    if scenario == "turn":
        return np.array([0., 0., .5, 0.], dtype=np.float32), waypoint
    point = np.asarray(state[:2])*16
    delta = WAYPOINTS[waypoint]-point
    if np.linalg.norm(delta) < .18:
        waypoint = (waypoint+1) % len(WAYPOINTS)
        delta = WAYPOINTS[waypoint]-point
    movement = delta/max(1e-6, np.linalg.norm(delta))
    angle = math.atan2(float(state[2]), float(state[3]))
    c, s = math.cos(angle), math.sin(angle)
    forward = c*movement[0]+s*movement[1]
    strafe = -s*movement[0]+c*movement[1]
    desired = 0.
    seconds = step*dt
    phase = seconds % 60
    combat_period = scenario == "combat" or (scenario == "mixed" and 25 <= phase < 50)
    if combat_period:
        candidates = [e[:2]*16 for e in entities(state) if e[5] >= .5]
        if candidates:
            target = min(candidates, key=lambda p: float(np.linalg.norm(p-point)))
            desired = math.atan2(target[1]-point[1], target[0]-point[0])
    elif scenario == "mixed" and 10 <= phase < 20:
        desired = (seconds-10)*.5
    error = (desired-angle+math.pi) % (2*math.pi)-math.pi
    turn = np.clip(error/(2.2*dt), -1, 1)
    # Two brief shots per minute exercise combat without deliberately clearing
    # the entire arena in the first seconds and calling the rest "active play".
    fire_phase = 5. if scenario == "combat" else 35.
    fire = combat_period and (fire_phase <= phase < fire_phase+dt*.9
                             or fire_phase+.30 <= phase < fire_phase+.30+dt*.9)
    return np.array([forward, strafe, turn, float(fire)], dtype=np.float32), waypoint


def build_action_tape(seed, scenario, steps, dt, shadow_reference=True, *, endless=False,
                      initialize_rgb=False, initial_image_size=128):
    """Record all controls before the model starts; no model feedback exists."""
    game = Game(seed=seed, endless=endless)
    survival_policy = None
    if endless and shadow_reference and scenario in {"explore", "combat", "mixed"}:
        from .prepare_training_data import SurvivalPolicy
        survival_policy = SurvivalPolicy(seed, 1/dt)
    tilemap, initial_state = game.map.copy(), game.state_vector().copy()
    initial_rgb = game.render(size=initial_image_size, overlays=False) if initialize_rgb else None
    states = [initial_state.copy()] if shadow_reference else None
    reference_events, actions = [], []
    waypoint, terminal_step = 0, None
    controller_state = initial_state.copy()
    for index in range(steps):
        if survival_policy is not None:
            action = survival_policy(game, index)
        else:
            action, waypoint = scripted_action(controller_state, scenario, index, dt, waypoint)
        actions.append(action)
        if shadow_reference:
            event = game.step(action, dt)
            controller_state = game.state_vector().copy()
            states.append(controller_state)
            reference_events.append(Game.event_vector(event))
            if terminal_step is None and game.done:
                terminal_step = index+1
        else:
            # Unqualified diagnostic tape: kinematic patrol planning only,
            # without physics, NPC updates or a reference correctness claim.
            angle = math.atan2(float(controller_state[2]), float(controller_state[3])) + action[2]*2.2*dt
            c, s = math.cos(angle), math.sin(angle)
            command = action[:2]/max(1., float(np.linalg.norm(action[:2])))
            controller_state[:2] += np.array([c*command[0]-s*command[1],
                                              s*command[0]+c*command[1]])*(2.4*dt/16)
            controller_state[2:4] = [s, c]
    tape = {"map": tilemap, "initial_state": initial_state,
            "actions": np.asarray(actions, dtype=np.float32),
            "reference_states": None if states is None else np.asarray(states, dtype=np.float32),
            "reference_events": None if not reference_events else np.asarray(reference_events, dtype=np.float32),
            "reference_terminal_step": terminal_step,
            "reference_completed": bool(shadow_reference),
            "endless": bool(endless),
            "controller": "survival_patrol_with_combat" if survival_policy is not None else "scenario_script",
            "tape_recorded_before_model_execution": True}
    if initial_rgb is not None:
        tape["initial_rgb"] = initial_rgb
    return tape


def _events_from_output(result):
    if "events" in result:
        event = result["events"]
        if isinstance(event, dict):
            return np.array([bool(event.get(name, False)) for name in EVENT_NAMES]), "structured_events"
        return event[0].detach().float().cpu().numpy(), "structured_events"
    return result["event_logits"][0].sigmoid().detach().float().cpu().numpy(), "independent_legacy_logits"


def _summary(values):
    if not values:
        return None
    return {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95))}


def evaluate_episode(model, metadata, args, seed):
    import torch
    dt = 1/float(getattr(model.config, "simulation_hz", 30))
    steps = max(1, math.ceil(args.seconds/dt))
    preparation_started = time.perf_counter()
    endless = bool(getattr(model.config, "endless", False))
    architecture = getattr(model.config, "architecture", "legacy_cnn")
    tape = build_action_tape(seed, args.scenario, steps, dt, not args.no_shadow_reference,
                             endless=endless, initialize_rgb=architecture == "vjepa2_ac",
                             initial_image_size=model.config.image_size)
    preparation_seconds = time.perf_counter()-preparation_started
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    # Discrete resources and a 9000-step clock must not accumulate in FP16.
    state_dtype = torch.float32 if getattr(model.config, "transition_mode", "residual_v0") == "event_v1" else dtype
    state = torch.tensor(tape["initial_state"], dtype=state_dtype, device=device)[None]
    grid = torch.tensor(tape["map"], dtype=torch.long, device=device)[None]
    monitor = RolloutMonitor(tape["map"], args.scenario, steps, dt, args.max_failures, endless=endless)
    render_oracle = Game(seed=seed, endless=endless) if args.render_every else None
    model_times, check_times, render_psnr, pose_errors, angle_errors = [], [], [], [], []
    render_times = []
    rendered, render_failures, render_oracle_rejections = 0, 0, 0
    matched_events, event_comparisons = 0, 0
    confusion = np.zeros((4, 4), dtype=np.int64)  # TP, FP, FN, TN for each event
    event_source = "unknown"
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        map_features = model.encode_map(grid)
        if architecture == "vjepa2_ac":
            initial_image = torch.as_tensor(tape["initial_rgb"], device=device).permute(2, 0, 1)[None]
            initial_image = initial_image.to(dtype=dtype).div(255)
            memory = model.initialize_from_image(initial_image, state, map_features)
        else:
            memory = model.initial_memory(1, device, dtype)
        synchronize()
        started = time.perf_counter()
        for index, action in enumerate(tape["actions"]):
            old = state[0].float().cpu().numpy().copy()
            action_tensor = torch.tensor(action, dtype=state_dtype, device=device)[None]
            synchronize()
            t0 = time.perf_counter()
            output = model.transition(state, action_tensor, map_features, memory)
            synchronize()
            model_times.append(1000*(time.perf_counter()-t0))
            state, memory = output["state"], output["memory"]
            current = state[0].float().cpu().numpy().copy()
            event_values, event_source = _events_from_output(output)
            reference = tape["reference_states"]
            ref_previous, ref_current = ((None, None) if reference is None else
                                         (reference[index], reference[index+1]))
            t0 = time.perf_counter()
            violations = monitor.observe(old, current, action, event_values, ref_previous, ref_current)
            if not bool(torch.isfinite(memory).all()):
                entry = {"code": "non_finite_memory", "message": "GRU memory contains NaN/infinity"}
                monitor.counts[entry["code"]] += 1
                if monitor.first_failure is None:
                    monitor.first_failure = {"step": index+1, "time_seconds": (index+1)*dt, **entry}
                violations.append(entry)
            check_times.append(1000*(time.perf_counter()-t0))
            if ref_current is not None and np.isfinite(current).all():
                pose_errors.append(float(np.linalg.norm((current[:2]-ref_current[:2])*16)))
                a0, a1 = math.atan2(current[2], current[3]), math.atan2(ref_current[2], ref_current[3])
                angle_errors.append(abs((a0-a1+math.pi) % (2*math.pi)-math.pi)*180/math.pi)
                matched_events += int(np.count_nonzero((event_values >= .5) == (tape["reference_events"][index] >= .5)))
                event_comparisons += 4
                predicted_flags, reference_flags = event_values >= .5, tape["reference_events"][index] >= .5
                confusion[:, 0] += predicted_flags & reference_flags
                confusion[:, 1] += predicted_flags & ~reference_flags
                confusion[:, 2] += ~predicted_flags & reference_flags
                confusion[:, 3] += ~predicted_flags & ~reference_flags
            if args.render_every and ((index+1) % args.render_every == 0 or index == steps-1):
                rendered += 1
                synchronize()
                render_started = time.perf_counter()
                rgb = model.render(state, output["features"], map_features)
                synchronize()
                render_times.append(1000*(time.perf_counter()-render_started))
                if not bool(torch.isfinite(rgb).all()):
                    render_failures += 1
                else:
                    try:
                        render_oracle.set_state_vector(current)
                        expected = render_oracle.render(model.config.image_size, overlays=False).astype(np.float32)/255
                        predicted = rgb[0].float().cpu().permute(1, 2, 0).numpy()
                        mse = float(np.mean((np.clip(predicted, 0, 1)-expected)**2))
                        render_psnr.append(float(-10*math.log10(max(1e-12, mse))))
                    except (ValueError, IndexError, OverflowError):
                        render_oracle_rejections += 1
            progress_every = getattr(args, "progress_every", 300)
            if progress_every and ((index+1) % progress_every == 0 or index == 0 or index == steps-1):
                print(json.dumps({"evaluation_step": index+1, "requested_steps": steps,
                                  "evaluation_seed": seed, "simulated_seconds": (index+1)*dt,
                                  "integrity_failures": sum(monitor.counts.values()),
                                  "respawns": monitor.respawn_count}), flush=True)
            if violations and (not args.continue_on_failure or not np.isfinite(current).all()):
                break
        synchronize()
        wall_seconds = time.perf_counter()-started
    report = monitor.report(tape["reference_completed"], tape["reference_terminal_step"])
    full_visual_samples = bool(rendered and not render_failures and not render_oracle_rejections)
    visual_pass = bool(full_visual_samples and args.min_render_psnr is not None
                       and render_psnr and min(render_psnr) >= args.min_render_psnr)
    event_metrics = {}
    for event_index, name in enumerate(EVENT_NAMES):
        tp, fp, fn, tn = map(int, confusion[event_index])
        event_metrics[name] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                               "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                               "note": "Null F1 means no positive reference/prediction examples; true negatives alone do not prove combat accuracy."}
    report.update({"seed": seed, "scenario": args.scenario, "requested_seconds": args.seconds,
                   "endless": endless, "controller": tape["controller"],
                   "world_architecture": architecture,
                   "engine_renders_for_initialization": int("initial_rgb" in tape),
                   "renderer_mode": getattr(model.config, "renderer_mode", "direct"),
                   "simulation_hz": 1/dt, "requested_steps": steps,
                   "wall_seconds": wall_seconds, "tape_preparation_seconds": preparation_seconds,
                   "tape_recorded_before_model_execution": True,
                   "engine_corrections": 0, "model_resets_during_rollout": 0,
                   "event_source": event_source,
                   "state_dtype": str(state_dtype), "network_dtype": str(dtype),
                   "state_only_evaluation": args.render_every == 0,
                   "rendered_sample_count": rendered, "render_nonfinite_count": render_failures,
                   "render_oracle_rejections": render_oracle_rejections,
                   "sampled_rgb_psnr_at_same_predicted_state": _summary(render_psnr),
                   "proposed_minimum_sampled_psnr": args.min_render_psnr,
                   "sampled_visual_gate_pass": visual_pass,
                   "visual_quality_validated": visual_pass,
                   "requested_active_pass": bool(report["state_integrity_active_requested_pass"] and visual_pass),
                   "300s_active_pass": bool(report["state_integrity_active_300s_pass"] and visual_pass),
                   "paired_trajectory_pose_error_tiles": _summary(pose_errors),
                   "paired_trajectory_heading_error_degrees": _summary(angle_errors),
                   "paired_event_bit_accuracy": matched_events/event_comparisons if event_comparisons else None,
                   "paired_events": event_metrics,
                   "transition_time_ms": _summary(model_times), "render_time_ms": _summary(render_times),
                   "checker_time_ms": _summary(check_times),
                   "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device)/1024**2 if device.type == "cuda" else None,
                   "limitations": ["Integrity and selected-pose RGB measurements are finite empirical tests, not a proof of generalization.",
                                   "Paired trajectories can diverge after combat branches; long-horizon pixel equality is not required.",
                                   "Sampled RGB compares the model with the source renderer at the SAME predicted valid state, without feeding that state back.",
                                   "No sampled RGB gate is accepted unless --min-render-psnr is explicitly configured.",
                                   "A copied/frozen model cannot qualify through elapsed time alone."]})
    return report


def main(argv=None):
    import torch
    from .models import load_world_checkpoint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=300.)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--scenario", choices=SCENARIOS, default="explore")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--render-every", type=int, default=30,
                        help="Render one diagnostic RGB every N steps; 0 means state-only and no visual pass")
    parser.add_argument("--no-render", action="store_true", help="Alias for --render-every 0")
    parser.add_argument("--max-failures", type=int, default=50,
                        help="Maximum stored failure examples; counts always include every checked violation")
    parser.add_argument("--progress-every", type=int, default=300,
                        help="Print real evaluation progress every N steps; zero disables progress logging")
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="Continue after finite invalid states; default stops at the first failed integrity step")
    parser.add_argument("--no-shadow-reference", action="store_true",
                        help="Skip paired source physics; diagnostic tapes then cannot qualify as active sessions")
    parser.add_argument("--min-render-psnr", type=float,
                        help="Optional proposed minimum PSNR for EVERY sampled same-state RGB; no universal threshold is implied")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds <= 0 or min(args.episodes, args.threads, args.max_failures) < 1 or args.render_every < 0 or args.progress_every < 0:
        parser.error("seconds/counts must be positive; render-every may be zero")
    if args.min_render_psnr is not None and not math.isfinite(args.min_render_psnr):
        parser.error("min-render-psnr must be finite")
    if args.no_render:
        args.render_every = 0
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(args.threads)
    model, metadata = load_world_checkpoint(args.world, args.device)
    model.eval()
    episodes = []
    for episode in range(args.episodes):
        result = evaluate_episode(model, metadata, args, args.seed+episode)
        episodes.append(result)
        print(json.dumps({"episode": episode+1, "checked_seconds": result["checked_simulated_seconds"],
                          "integrity_pass": result["integrity_pass"],
                          "qualifying_active_session": result["qualifying_active_session"],
                          "first_failure": result["first_failure"]}), flush=True)
    report = {"evaluation": "free_neural_rollout_integrity_and_active_coverage", "format_version": 1,
              "checkpoint": str(args.world.resolve()), "training_steps": metadata.get("training_steps", 0),
              "quality_status": "Empirical evaluation only; trained accuracy, requested-duration stability and target FPS require measurement.",
              "device": args.device, "transition_mode": getattr(model.config, "transition_mode", "residual_v0"),
              "endless": bool(getattr(model.config, "endless", False)),
              "renderer_mode": getattr(model.config, "renderer_mode", "direct"),
              "episodes": episodes, "episode_count": len(episodes),
              "target_seconds": args.seconds,
              "target_steps": max(1, math.ceil(args.seconds/(1/float(getattr(model.config, "simulation_hz", 30))))),
              "integrity_pass_count": sum(e["integrity_pass"] for e in episodes),
              "qualifying_active_session_count": sum(e["qualifying_active_session"] for e in episodes),
              "state_integrity_active_requested_pass_count": sum(e["state_integrity_active_requested_pass"] for e in episodes),
              "requested_active_pass_count": sum(e["requested_active_pass"] for e in episodes),
              "state_integrity_active_300s_pass_count": sum(e["state_integrity_active_300s_pass"] for e in episodes),
              "300s_active_pass_count": sum(e["300s_active_pass"] for e in episodes),
              "generalization_note": "Current seeds vary trajectories/animation on a fixed arena; new-map generalization is not tested."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"Saved evaluation: {args.output}")
    return report


if __name__ == "__main__":
    main()
