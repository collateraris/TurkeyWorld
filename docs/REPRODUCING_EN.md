# Reproducing TurkeyWorld experiments

[README](../README.md) · [Русский: detailed walkthrough](REPRODUCING_RU.md) · [published architecture snapshot](architecture-snapshot-2026-11-15.json)

## What can be reproduced

There are three distinct tasks:

1. **Fresh baseline:** generate new Source data and train an initial codec/SR/teacher-world with no old game-model weights.
2. **Reconstruct the later experimental stack:** extend that baseline through RSSM/expert migrations, renderer training and the RRDB memory experiments.
3. **Replay the published recordings:** load pinned world8873, renderer and SR7000 checkpoints.

**The repository does not currently offer a single clean-room command that trains the entire late RSSM/RF3/RRDB stack from an empty directory.** `fresh_pipeline` provides the historical starting route. Later trainers are continuation stages with parent checkpoints, manifests and SHA contracts. `ordered_image_pipeline` also requires existing parents; it does not manufacture them.

The instructions distinguish executable baseline commands from later-stage reconstruction templates. A new training run is not expected to recover identical weights or metrics. No full multi-day clean-room reconstruction was executed while writing this guide.

## 1. Freeze code, assets and environment

Use a separate checkout/worktree and new run directories. Record `git rev-parse HEAD`, `git diff`, Python/Torch/CUDA versions and the NVIDIA driver. A Git commit alone does not reproduce uncommitted changes present at recording time. The configuration snapshot records its documentation revision.

Keep `assets/turkey_atlas.png` and `assets/gun_atlas.png`; their hashes are in the snapshot. These are actual pixel priors used by the hybrid SR, so changing them changes the experiment. Git ignores `training_runs`, datasets, `.pt` and `.npz`: cloning the repo does not retrieve historical checkpoints or data. Budget disk space for data, full model/Adam checkpoints and pinned snapshots.

Ubuntu24.04/NVIDIA setup:

```bash
bash scripts/setup_ubuntu.sh --dry-run
bash scripts/setup_ubuntu.sh
PY=.venv-ubuntu/bin/python
"$PY" -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'
command -v ffmpeg
command -v ffprobe
```

The installer uses Python3.12, Torch2.11/torchvision0.26 CUDA12.8 and pinned application packages. Install the driver separately. Setup does not start training or clear stop files. LPIPS may download loss-network weights on first use; these are not pretrained TurkeyWorld weights. CPU is useful for unit tests, whereas training is GPU-oriented. Find FFmpeg in PATH rather than assuming `/usr/bin/ffmpeg`; the recording workstation used Anaconda binaries.

## 2. Generate fresh Source training data

Run from the repository root, choosing an unused directory:

```bash
RUN=training_runs/reproduce_clean_01
PY=.venv-ubuntu/bin/python
"$PY" -m turkeyworld.prepare_training_data "$RUN/data" \
  --rgb-train 4000 --rgb-val 256 --clip-frames 32 \
  --size 32 --hr-size 128 \
  --long-train 64 --long-val 16 --long-frames 450 \
  --hz 30 --seed 173 --workers 4
```

This is an example baseline inventory, not the exact historical counts of every later phase. The fresh supervisor pins128000 training transitions, requiring4000×32 short clips. HR must be true Source128, not enlarged Source32. State/action/map/RGB/mask data and long validation episodes are retained.

Inspect `data/status.json` and manifests; verify completion, `BEHAVIOR_VERSION`, `single_enemy_v1` and disjoint TRAIN/VAL conditions. Do not change code/assets/seeds within a pinned inventory. Later stages additionally need navigation, quiet-enemy, combat and static datasets covering near/mid/far NPCs, empty views, turns, corners, blocked motion and axis sliding. Their quotas and hashes belong in new plans.

## 3. Start the historical baseline from zero

```bash
bash scripts/run_training.sh --module turkeyworld.fresh_pipeline \
  --run-dir "$RUN/pipeline" --data-dir "$RUN/data" \
  --codec-run-dir "$RUN/codec" --sr-run-dir "$RUN/sr" \
  --world-run-dir "$RUN/world" --monitor-control-dir "$RUN/monitor" \
  --image-size 32 --device cuda --target-seconds 6 \
  --batch 4 --gradient-accumulation-steps 1 \
  --codec-batch 8 --sr-batch 8 --sr-steps 2000
```

The supervisor starts fresh VQGAN, original x4 SR and teacher-world/renderer stages. The codec has a quality gate; a child `--max-steps 0` means continue until that criterion, not zero training. Runtime/completion is not guaranteed. Weakening a gate does not constitute passing the original criterion.

**This baseline is not yet the three-RSSM/RRDB wall-memory system.** It supplies a trained starting teacher for the historical migration route. Child commands are saved in pipeline `*_command.json`; stage directories contain counters/logs/checkpoints.

```bash
"$PY" -m training_dashboard.local_server --run-dir "$RUN/monitor" --port 8790
```

## 4. Reconstruct the current dynamics topology

The sequence below is a dependency plan, not an automated shell script. Each operation needs a compatible completed parent and a new output/receipt. Do not edit `training_steps` to pretend a random checkpoint reached a historical boundary.

| Order | Implementation | Purpose |
|---|---|---|
| 1 | `rssm_migration.initialize_rssm_checkpoint` | Initialize Player RSSM from the trained teacher, then train/evaluate it |
| 2 | `dual_rssm_migration.prepare_dual_rssm_continuation` | Add/train Enemy RSSM |
| 3 | `enemy_rssm_resize.prepare_enemy_rssm_resize` | Match Enemy h1344/hidden1408 if necessary |
| 4 | `rssm_fusion_migration.prepare_rssm_fusion_continuation` | Add shared d256/top-2 MoE |
| 5 | `rssm_combat_memory_migration.prepare_combat_memory_continuation` | Add Combat h1344/hidden1392 |
| 6 | `rssm_five_migration.prepare_rssm_five_continuation` | Add five private role experts |
| 7 | `rssm_five_expanded_migration.prepare_expanded_five_continuation` | Expand each expert to1,005,088 parameters |
| 8 | `rssm_spawn_conditioning_migration.prepare_spawn_conditioning_continuation` | Add current spawn conditioning |

These are Python functions in `turkeyworld/`; not all have CLI entrypoints. Inspect signatures/preconditions. An example of **only the first migration**, after a compatible teacher actually exists:

```python
from pathlib import Path
from turkeyworld.rssm_migration import initialize_rssm_checkpoint
initialize_rssm_checkpoint(
    Path("training_runs/reproduce_clean_01/world/checkpoints/teacher.pt"),
    Path("training_runs/reproduce_clean_01/rssm_initial.pt"),
    seed=173,
)
```

This is not a pretrained download. Adding random branches does not train them. Some migrations require optimizer/training ledgers from specific earlier phases, not merely model tensors; an explicit adapter/new experiment plan is needed when those contracts do not match. Do not disable checks or fabricate ledgers.

Subsequent trainers include `train_rssm_dynamics`, `train_rssm_moe` and `train_rssm_motion_only`. Create new data/batch plans and satisfy their bind/activation contracts. The present target is H180/100, MAIN16=8 NAV+8 quiet NPC. Previous ladders are historical, not the current round definition. The architecture JSON describes topology, not trained weights/Adam.

## 5. Rebuild the image chain

Order: VQGAN encoder/quantizer/decoder → trained teacher renderer → distilled renderer/Base → five image roles → expanded Base → RF3 → decoder overlay. Keep source-paired geometry/foreground labels and a separate VAL panel.

```bash
"$PY" -m turkeyworld.distill_renderer --help
"$PY" -m turkeyworld.train_renderer_refine --help
"$PY" -m turkeyworld.train_base_geometry --help
"$PY" -m turkeyworld.train_decoder_rf_chain --help
"$PY" -m turkeyworld.ordered_image_pipeline --help
```

The ordered pipeline requires `--base-run`, `--world`, `--source-recipe-parent`, `--decoder-parent`, `--sr-parent` and other prepared inputs. Its historical stage budgets do not initialize missing parents. Distillation CLI accepts solver1/4; do not invent `--solver-steps 3` there. RF3 is a later explicit bundle/training chain, not a free change to any renderer's integration count.

Pin the completed image bundle with format/config/SHA and actual Euler3 solver. Decoder adaptation updates only decoder/post-quant layers, preserving encoder/quantizer. NPR32317/Base31709/decoder4125 are independent historical artifact clocks, not required update targets for a new run.

## 6. Prepare SR pairs and reconstruct RRDB

Generate true Source32/128 pairs, then render their inputs through your frozen neural renderer:

```bash
"$PY" -m turkeyworld.sr_neural_pairs_data \
  --output "$RUN/source_pairs" --source "$RUN/data" \
  --train-count 8192 --val-count 1024 --workers 4 --seed 173

"$PY" -m turkeyworld.cache_sr_neural_pairs \
  --source-root "$RUN/source_pairs" --output "$RUN/neural_pairs" \
  --world "$RUN/rssm_world.pt" --renderer "$RUN/renderer.pt" \
  --device cuda --batch 8 --seed 173
```

The cache command is a template for a stage **after compatible model files exist**. Pair collection requires sufficient near/mid/far/guard inventory; add genuinely different Source conditions if the baseline collection does not cover them. Do not duplicate frames to satisfy counts. Unqualified experimental codecs require explicit `--allow-unqualified-codec` and remain experimental.

A fresh RRDB experiment needs new RGB weights/Adam, trained enemy detection/box/phase/visibility heads, then weapon memories, depth/UV and column-material/side heads. Preserve embedded asset hashes and use held-out panels rather than fitting the demonstration screenshots.

**Automation boundary:** current `train_rrdb_material_sr` is a continuation trainer reading `run-dir/assets/source.pt`. Its paired-data/VAL paths still reference `image_chain_20261114/data/paired` and `sr_background_detail_20261114/training_plan.json`. `train_sprite_heads_sr` also depends on historical data/plans. Clean-room reconstruction requires parameterizing these paths and creating new plans. Running these commands against absent historical directories is not a complete fresh-training recipe.

Reproduce learning regimes deliberately: main G2e−5/D5e−6, separate added-head groups1e−4, GAN≤0.002 and5% readout-gradient ratio. Evaluate false-green/depth/material accuracy, sprite dropouts and temporal behavior separately. Source geometry stays label-only.

## 7. Policy and serial training

`train_dreamer_policy` has historical TRAIN_DIRS/HEAD_EVENT_DIR defaults. Point them at your own navigation/quiet/combat corpora; `--world-source` alone does not replace the data inventory. Use finite update budgets, fixed seeds and separate VAL.

`dreamer_motion_cycle` alternates policy-controlled Source collection, H180/100 world updates and actor/critic updates. Explicitly provide new `--run-dir`, `--world-run`, `--policy-run`, `--motion-plan` and `--initial-world`. Prepare compatible full checkpoints before starting it. World/graphics freeze during policy training; graphics freeze during motion-only dynamics.

## 8. Replay and record completed models

Pin completed world/renderer/SR snapshots. The following use example names for **your compatible completed artifacts**:

```bash
"$PY" -m turkeyworld.compare_play \
  --world "$RUN/rssm_world.pt" --renderer-checkpoint "$RUN/renderer.pt" \
  --sr "$RUN/sr.pt" --rf-euler-steps 3 --device cuda \
  --display-size 1024 --smooth --initial-environment navigation
```

Add the experimental codec opt-in only when required and intentionally accepted. Enter starts, T switches environment, R resets both branches, Esc exits. Check autonomous mode/shared initialization/identical controls/no Source corrections. A baseline teacher is not dynamics8873.

```bash
"$PY" -m turkeyworld.record_paired_cuts --profile navigation-six \
  --output-dir "$RUN/video_navigation" \
  --world "$RUN/rssm_world.pt" --renderer "$RUN/renderer.pt" --sr "$RUN/sr.pt" \
  --ffmpeg "$(command -v ffmpeg)"

"$PY" -m turkeyworld.record_quiet_chase --near --tracking \
  --output-dir "$RUN/video_enemy" \
  --world "$RUN/rssm_world.pt" --renderer "$RUN/renderer.pt" --sr "$RUN/sr.pt" \
  --ffmpeg "$(command -v ffmpeg)"
```

Recorders explicitly opt into experimental codecs. Quiet-chase records16 trials and selects six complete cuts by its disclosed score. Inspect candidate/report JSON, HP/events, all starts, ffprobe and full decode; curation does not guarantee stable sprites. A36s/30fps montage contains1080 frames. Publish selection details and remaining errors with the video.

## 9. Resume and acceptance

Keep model/config, named Adam/group order, global step, Python/NumPy/Torch/CUDA RNG, private samplers, manifests/plans/SHA, migration ledgers and image bundles. `state_dict` alone is insufficient for exact resume. Stop only at complete optimizer boundaries; do not automatically delete stop files.

Compare equal VAL/H/actions/seeds: XY drift, heading, wall crossings/stalls, NPC velocity/direction, HP/events/spawn, image SSIM/MAE, material/false-green and flicker. A changed TOTAL objective is not directly comparable to its predecessor.

```bash
PYTHONPATH=tests:. "$PY" -m unittest \
  test_rssm_first_contact test_rssm_motion_only_training \
  test_rrdb_material_sr test_wall_texture_memory_sr test_weapon_memory_sr \
  test_record_paired_cuts test_record_quiet_chase
```

Command syntax/dependencies were checked against source. Full late-stage clean-room automation remains an explicit packaging task; documentation does not bypass historical checkpoint guards or claim that it has already been validated end to end.
