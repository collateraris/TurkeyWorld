# TurkeyWorld: a neural game world

[English](README.md) | [Русский](README_RU.md)

TurkeyWorld combines a playable first-person reference game with an experimental
action-conditioned neural world model. The current experiment trains a compact
V-JEPA 2-AC adaptation with a **500M Flow Matching teacher**, then a **22.48M
Rectified Flow student** intended for interactive play. The next stage trains a
shared VQGAN codec from scratch before continuing both models in its latent space.

**Five minutes of correct neural gameplay is a qualification target, not an
achieved result implied by a checkpoint.** Training progress, image similarity,
and autonomous gameplay checks are reported separately. The original game is
playable independently of any trained neural weights.

## Installation and the reference game

Run commands from the repository root. These PowerShell examples use a local
virtual environment; an existing compatible Python environment also works.

```powershell
python -m venv .venv
$twPython = Join-Path $PWD '.venv\Scripts\python.exe'
& $twPython -m pip install -r requirements-ml.txt
& $twPython -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

`requirements.txt` contains NumPy, Pillow, and pygame. `requirements-ml.txt` adds
PyTorch. GPU training requires a CUDA-capable PyTorch installation; verify the
CUDA check before passing `--device cuda`. CPU execution is supported for small
tests and inference, but the 500M training experiment is designed for a GPU.

```powershell
& $twPython -m turkeyworld.play --size 128 --display-size 1024
```

Controls: **WASD** to move, **Q/E or left/right arrows** to turn, **Space or left
mouse button** to fire, **Esc** to exit. The reference game also supports **R**
to restart. The default reference mode has unlimited ammunition and endless
enemy respawns, with eight persistent entity slots and cumulative kills.
`--classic` selects the earlier finite-enemy mode.

The reference game uses the original engine. It is separate from neural
playback, whose state is advanced by a learned model.

## Architecture and what the model learns

The architecture adapts principles from the MIT-licensed
[official V-JEPA 2 repository](https://github.com/facebookresearch/vjepa2) and
[V-JEPA 2 paper](https://arxiv.org/abs/2506.09985): visual patch latents,
action/state conditioning, a predictive latent space, and a separate EMA target
encoder during training. **Official Meta pretrained weights are not loaded.**
This is a smaller game-specific implementation, not the official large model.

The table below describes the preserved RGB checkpoint profile. The VQGAN
continuation and its updated parameter counts are documented below.

| Component | Teacher | Selected RGB student |
|---|---:|---:|
| Inference generator parameters | **500,044,741** | **22,475,013** |
| Visual latent width | 512 | 320 |
| Video encoder blocks | 76 | 8 |
| Action-conditioned predictor blocks | 58 | 5 |
| Attention heads | 8 | 5 |
| Game readout MLP ratio | 9 | 4 |
| RGB generator | Conditional Flow Matching | Rectified Flow |
| Image sampling | 12 full Heun steps | 4 Euler steps |
| Student RF transformer | — | Width 192, depth 4, heads 4 |
| Separate EMA encoder parameters | 239,975,936 | 10,110,400 |

EMA encoders and GAN discriminators are training components; they are excluded
from inference generator counts. The selected student is defined explicitly in
[student_profile.py](turkeyworld/student_profile.py). The earlier 11,747,077
parameter student remains the default for legacy checkpoint compatibility.
The current 250M/500M experiment runners select the new 22.48M profile.
Parameter counts do not establish inference FPS.

At 128×128 resolution, 16×16 patches produce 64 visual tokens per frame. Each
temporal block packs an action token, a 90-value state token, and 64 visual
tokens. Runtime memory holds four blocks: 264 tokens. Attention is bidirectional
within a frame and causal between frames, with temporal/height/width RoPE.
The video encoder is causal too: encoding a clip does not expose its future
images to earlier latents.

At the beginning of a segment, the model receives a source RGB context, map,
and initial state. After that, the runtime carries **its own predicted visual
memory and state**, receiving only new controls. Learned map features persist
and are read globally and near the player and all eight entities. Off-camera
objects remain in their persistent slots; the camera does not delete them.

This is a **hybrid** world model. Neural heads predict motion, shot targets,
attacks, and respawn locations. Explicit PyTorch rules update health, life
flags, timers, heading, ammunition, and cumulative kills. Motion also obeys
explicit command-direction and speed bounds. Respawns use a map-derived
floor/distance mask. These disclosed rules are not presented as
learned physics. Collision avoidance and combat accuracy still have to be
learned and tested; the original engine does not repair neural execution.

The 500M teacher grows from a trained 250M checkpoint. Every original parameter
value is retained, including the original slices of widened readout matrices.
Added residual transformer blocks initially compute identity; additional
readout channels initially have zero output columns. This preserves the
initial function while adding trainable capacity. Gradient checkpointing
reduces training activation memory. Migration code is in
[vjepa2_ac.py](turkeyworld/vjepa2_ac.py).

## Data: targets, provenance, and collection

The current target is **5,000,000 base RGB training transitions**:
156,250 clips × 32 transitions. An additional 256 model-visited clips contribute
8,192 transitions, giving a planned short-training total of **5,008,192**.
The fixed RGB validation split contains 512 clips, or 16,384 transitions.
These are plan sizes; actual completed counts come from dataset reports.

Long recordings target 256 complete training episodes and 32 complete validation
episodes of 9,000 transitions each. Recordings stop on natural player death;
partial episodes are retained and do not count toward the full-length quota.
The collector adds independent episodes until genuine full-length targets are
met. It does not concatenate resets or heal the player inside a recording.

The corpus includes ordinary movement, wall contact, combat after the first
action, deliberate misses, kills, respawns, delayed attacks, and returning to
previously seen objects. Valid perturbed **reference** states are training-only
augmentation; they are not claimed to be model-visited states.

Model-visited collection is a separate procedure:

1. Initialize a frozen previous 125M teacher from a randomized valid source state
   and one source image; record a control tape before its neural rollout.
2. Run its own state and memory forward without source corrections or new RGB.
3. After that rollout finishes, select its latest sampled valid own state.
4. Start a separate oracle branch at that state and record the true continuation
   for training. Invalid states are skipped without projection or repair.

Oracle results never feed back into the completed main neural rollout. The
collector records the teacher fingerprint, visited step, original own state,
and branch initialization. A collection with no valid states is reported as
incomplete, rather than replaced by reference augmentation.

Each RGB NPZ contains aligned states, actions, clean RGB frames, events, hit
IDs, attack/respawn flags, depth, visibility, the map, timestep, seed, and exact
initialization metadata. The action at index `t` belongs to transition
`state[t] → state[t+1]`; observations have one more entry than actions.
Train and validation use disjoint seed ranges.

Start or resume CPU data expansion in one terminal:

```powershell
$twPython = Join-Path $PWD '.venv\Scripts\python.exe'
& $twPython -m turkeyworld.data_expansion training_runs/vjepa250_20261109/data `
  --source training_runs/flow_20261108/data `
  --rgb-train 156250 --rgb-val 512 --clip-frames 32 --size 128 `
  --long-train 256 --long-val 32 --long-frames 9000 --workers 8
```

Source reuse uses hardlinks where possible, otherwise copies. An absent reuse
source does not prevent generating the base engine corpus. Target growth
appends immutable jobs and new seeds; completed files and the frozen validation
split are preserved. Repeat the same command to resume.

The additional model-visited clips require the previous trained 125M checkpoint:

```powershell
& $twPython -m turkeyworld.collect_model_states `
  --checkpoint training_runs/flow_20261108/checkpoints/teacher_for_student_vjepa125.pt `
  --output training_runs/vjepa250_20261109/data/rgb/train `
  --count 256 --seed 20000000 --steps 32 --max-model-steps 32 --device cuda
```

Run this GPU collection before training or while the GPU is available.
The current prepared corpus already includes
its completed 256-clip manifest; rerunning the same collection validates and
reuses matching files.

The active 500M run shares the same physical corpus through a Windows directory
junction. Create the directories and junction only if they are absent:

```powershell
$sharedData = Join-Path $PWD 'training_runs\vjepa250_20261109\data'
$runDirectory = Join-Path $PWD 'training_runs\vjepa500_20261109'
$linkedData = Join-Path $runDirectory 'data'
New-Item -ItemType Directory -Path $sharedData -Force | Out-Null
New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
if (-not (Test-Path -LiteralPath $linkedData)) {
  New-Item -ItemType Junction -Path $linkedData -Target $sharedData | Out-Null
}
```

Training streams episodes with a bounded 512MiB LRU cache, refreshing the
directory approximately every 30 seconds. CPU collection can continue while
the model trains on completed files; the runner does not wait for all 5 million
transitions once its minimum corpus and model-visited prerequisites are ready.

Reports include `data/status.json`, `data/progress.jsonl`, and
`data/rgb/train/model_visited_manifest.json`. Use completed counts and actual
event coverage from these reports when assessing the corpus.

## Shared VQGAN codec for Teacher and Student

The codec run uses the existing game RGB corpus, with validation episodes kept
separate. Its encoder, codebook, decoder, and PatchGAN discriminator start from
random weights. **No pretrained codec or discriminator weights are loaded.**
The frozen pretrained LPIPS/VGG network supplies a perceptual training loss only;
it is absent from inference. This compact implementation adapts the
[MIT-licensed CompVis/taming-transformers design](https://github.com/CompVis/taming-transformers)
and does not support official VQGAN checkpoints.

The selected codec has **4,763,731 parameters** across encoder, codebook, and
decoder, with 1,488 codes of width 16. It maps 128×128 RGB to a 32×32 latent grid.
Its discriminator is counted separately. Source pixels are legitimate codec
inputs: this stage measures image reconstruction, independently of world prediction.

```powershell
& $twPython -m pip install -r requirements-vqgan.txt
& $twPython -m turkeyworld.train_vqgan `
  --run-dir training_runs/vqgan_20261109 `
  --data-dir training_runs/vjepa250_20261109/data `
  --world-source-run training_runs/vjepa500_20261109 `
  --batch 8 --disc-start 1000 --validate-every 500 --continue-world
```

Batch 8 means eight adjacent frame pairs. Reconstruction, quantization,
perceptual, edge, and temporal losses train from the beginning; adversarial
training starts at step 1,000. Every 500 updates, the dashboard displays actual
held-out **Source / reconstruction / absolute RGB difference ×4** images and
reports codebook usage, perplexity, and temporal reconstruction error.

Training starts after checking 8,192 clips, then indexes another 2,048 every
discovery interval until the full corpus is available. `--initial-files 0`
requests full indexing before the first step. The monitor distinguishes indexed
clips, all files on disk, and actual RGB examples consumed; none is an epoch count.

The fixed panel contains 128 validation pairs, or 256 frames. At least 95% of
frames must jointly reach PSNR ≥40 dB, SSIM ≥0.99, and MAE ≤1/255, with finite
outputs and at least three active codes. Qualification requires **three
consecutive passing checks**, measured in FP32. Completion time remains unknown
until the gate passes. A codec checkpoint alone does not qualify the world model
for five minutes of play or establish its FPS.

After qualification, `codec_ready.json` records the frozen export
`checkpoints/codec_for_renderer.pt`. With `--continue-world`, the controller
migrates and resumes the preserved 500M world checkpoint in
`training_runs/vjepa500_vqgan_20261109`. Both Teacher Flow Matching and Student
Rectified Flow generate latents decoded by this same frozen codec. Their
inference totals, including the codec once per model, are **504,857,752** and
**22,475,096** parameters. The selected latent Student uses width 320, four
encoder blocks, five predictor blocks, readout ratio 5, and an RF transformer
of width 192, depth 4, and hidden width 736. The earlier pixel-space Student
checkpoint remains preserved separately.

Codec checkpoints preserve both optimizers and random state; rerun the same
command to resume. The controller also preserves the compatible teacher world
optimizer, EMA encoder, and completed world-training steps. It starts a detached
worker and releases the codec's GPU training resources. The dashboard follows
`active_run.json` automatically. A `stop_training` file in the codec run stops
the currently active world worker too; remove stop markers before resuming.

To resume an already migrated latent world job directly:

```powershell
& $twPython -m turkeyworld.train_vq_world `
  --run-dir training_runs/vjepa500_vqgan_20261109 `
  --codec-checkpoint training_runs/vqgan_20261109/checkpoints/codec_for_renderer.pt `
  --control-run-dir training_runs/vqgan_20261109
```

The codec reconstruction panel permits its Source input. The later
`free_rollout` world preview receives initial context and controls only;
future ground-truth images and states remain evaluation targets.

## Train the preserved RGB 500M →22.48M experiment

The current 500M workflow is a **checkpoint migration and continued-training
workflow**. If `checkpoints/teacher_vjepa500_initial.pt` does not exist, it
requires a trained 250M source checkpoint. The source checkpoints and generated
datasets are not tracked in Git. The 500M CLI does not currently provide a
single-command random-initialization bootstrap for a fresh checkout.
Reference gameplay and base engine-data generation work without these weights.

After the shared data path is ready, run this in a separate terminal:

```powershell
$twPython = Join-Path $PWD '.venv\Scripts\python.exe'
& $twPython -m turkeyworld.train_500m `
  --run-dir training_runs/vjepa500_20261109 `
  --source training_runs/vjepa250_20261109/checkpoints/teacher250_world.pt `
  --device cuda --batch 1 --cache-mb 512 `
  --pretrain-steps 6000 --world-repair-steps 3000 --world-patience 6 `
  --context 2 --horizon 4 --chunk 16 `
  --student-steps 16000 --repair-steps 2000
```

Training separates two streams. The **main stream** runs autoregressively on
its own predictions. Recorded future states and EMA visual latents supervise
that trajectory without being substituted into it. A **separate auxiliary
stream** queries exact recorded poses and past observations, so its motion,
combat, spawn, RGB, and flow labels describe the state actually queried.
Future source observations used by this training-only branch never enter the
main stream. Deployment, previews, and qualification use the autonomous stream.

The EMA encoder supplies stop-gradient JEPA targets. Variance/covariance
regularization limits collapsed representations. The image generator learns
conditional velocity fields; it is not merely a decoder named Flow Matching.
GAN training uses actual sampled final RGB, a conditional two-scale
discriminator, hinge loss, feature matching, and lazy R1. The auxiliary image
query uses a known target pose for training; autonomous image-quality checks
still evaluate the model's own future trajectory.

The stage sequence is:

1. **World dynamics:** 6,000 updates with the renderer frozen and no GAN.
   A held-out dynamics gate measures autonomous position/motion error, hit
   F1/recall, attack F1, collisions, and positive-example coverage. The current
   gate panel has 128 windows, or 512 future transitions.
2. **World repair if needed:** 3,000-update rounds. Six distinct gate checks
   without a cumulative 1% score improvement trigger a plateau stop. A failed
   gate remains recorded as failed; the best world checkpoint is restored
   before handing the teacher to the remaining pipeline.
3. **If the dynamics gate passes:** train the long teacher curriculum with
   horizons 32/128/512/2048/9000, totaling 14,888 transitions. TBPTT detaches
   graphs while carrying the numerical state and visual memory unchanged.
4. **Teacher RGB refinement:** when the world gate passed, tune full Flow
   Matching sampling with GAN until the visual target or visual plateau.
   The fixed held-out image panel contains 64 windows: 2 context frames and 4
   own future predictions each, totaling 256 evaluated future images.
5. **Freeze the selected teacher:** restore its actual selected weights and
   run independent `explore` and `mixed` evaluations of 9,000 transitions each.
6. **Student:** 16,000 updates using the selected 22.48M V-JEPA/RF profile,
   teacher distillation, and on-the-fly reflow every fourth update. Reflow
   uses the same initial noise for teacher endpoints and student targets.
7. **Student long curriculum and qualification:** if the two independent
   five-minute checks fail, continue with 2,000 short updates, another long
   curriculum, and two fresh-seed evaluations. Repeat until qualification
   passes. The teacher remains frozen throughout student repairs.

Visual qualification requires PSNR≥35 dB, SSIM≥0.98, and MAE≤2/255 together on
at least 95% of future images, acceptable aggregate SSIM/MAE, and three passing
checks in succession. Metrics use the full sampled FP32 renderer with TF32
disabled. Visual plateau uses six distinct checks without a cumulative 0.5%
improvement, checked every 500 refinement updates. Plateau is an alternate
handoff reason, not a quality pass.

The key implementation files are [train_500m.py](turkeyworld/train_500m.py),
[train_250m.py](turkeyworld/train_250m.py),
[aligned_training.py](turkeyworld/aligned_training.py),
[vjepa_long_rollout.py](turkeyworld/vjepa_long_rollout.py), and
[dynamics_gate.py](turkeyworld/dynamics_gate.py).

## Monitor, stop, and resume

```powershell
& $twPython -m training_dashboard.local_server `
  --run-dir training_runs/vjepa500_20261109 --port 8765
```

Open **http://127.0.0.1:8765**. The dashboard shows current model training and
ongoing data generation separately, including real counts, phase-specific loss
curves, dynamics failures/coverage, parameter counts, EMA/discriminator size,
checkpoints, and independent gameplay results. Missing metrics remain missing.

**Every 500 global optimizer updates**, `preview.png` is refreshed with the
initial context, future Source images, and the model's own predictions. Future
Source is displayed for comparison and does not correct the rollout. An
optimizer update may represent several simulated transitions in TBPTT.

Request a graceful stop at the next safe checkpoint boundary:

```powershell
New-Item -ItemType File -Path training_runs/vjepa500_20261109/stop_training -Force
```

To resume, remove the request and repeat the same training command:

```powershell
Remove-Item -LiteralPath training_runs/vjepa500_20261109/stop_training
```

Weights, optimizer, EMA, phase progress, and carried long-rollout state are
checkpointed. Student continuation has its own durable progress record. Do not
launch a duplicate trainer for the same run directory. The data-expansion
process has separate progress; pausing the trainer does not erase its files.

On Windows, the training runner requests that the computer stay awake while
allowing the display to sleep. **Win+L** can lock the screen without stopping
an already running background worker. Explicit sleep, shutdown, or signing out
still interrupts it. The guard does not change the power plan. An existing
worker can also be protected through the command below, provided its launcher
has written a valid PID file. The training runner itself does not create
`training.pid`; its built-in awake guard already protects a foreground run.

```powershell
& $twPython -m turkeyworld.power_guard `
  --pid-file training_runs/vjepa500_20261109/training.pid
```

The terminal commands above run in the foreground. Keep that terminal open
when locking the screen. To launch the default experiment as a hidden background
worker, use the following **instead of starting a second foreground trainer**:

```powershell
@'
from pathlib import Path
import subprocess
import sys

run_dir = Path("training_runs/vjepa500_20261109")
run_dir.mkdir(parents=True, exist_ok=True)
command = [sys.executable, "-m", "turkeyworld.train_500m", "--run-dir", str(run_dir)]
with (run_dir / "worker_stdout.log").open("ab") as output, (run_dir / "worker_stderr.log").open("ab") as errors:
    worker = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=errors,
                              creationflags=subprocess.CREATE_NO_WINDOW, close_fds=True)
(run_dir / "training.pid").write_text(str(worker.pid), encoding="utf-8")
print("Training PID:", worker.pid)
'@ | & $twPython -
```

This uses the runner defaults and the existing source/data prerequisites.
Add explicit CLI arguments to `command` if overriding the defaults. Logs and
the PID remain in the run directory; use the same `stop_training` request for
a graceful stop. A locked screen is separate from closing a process or console.

Useful run artifacts include:

- `status.json`, `metrics.jsonl`, `preview.png`, and `student_plan.json`.
- `checkpoints/teacher_vjepa500_initial.pt`, `teacher500_world*.pt`,
  `teacher500_vjepa_temporal.pt`, and `teacher_quality_500m*.pt`.
- `teacher_handoff_vjepa500.json` and
  `checkpoints/teacher_for_student_vjepa500.pt`.
- `checkpoints/student_vjepa500_distill.pt` and `student_vjepa_temporal.pt`.
- `dynamics_gate_*.json`, `teacher_world_progress500.json`, and
  `student_continuation_vjepa.json`.
- `teacher_vjepa_evaluation*.json` and `student_vjepa_evaluation*.json`.

## Timing, neural playback, and the five-minute check

An RTX4090 training benchmark with batch 1 measured approximately **5.08 s** per
500M world update and **5.89 s** for full Heun/GAN training, with a peak of
**13.83 GiB**. The initial 6,000-update phase has a provisional **8–10 hour** budget
plus validation; actual run speed and dashboard ETA take precedence.
CPU collection initially measured 30.37 clips/s, giving an approximately 80-minute
estimate and 64 GiB of additional storage for the 5M-transition expansion.
These are measurements/estimates from this experiment, not hardware promises.
Neither they nor the earlier student benchmark establish FPS for the new 22.48M
student. **Time until valid five-minute gameplay is unknown** because teacher
plateaus and repeated student repairs are not bounded by a completion deadline.

When the corresponding checkpoint exists, launch neural playback with:

```powershell
& $twPython -m turkeyworld.neural_play `
  --world training_runs/vjepa500_20261109/checkpoints/student_vjepa_temporal.pt `
  --device cuda --display-size 1024
```

The runtime uses one original-engine initialization image, then discards the
engine. Subsequent motion, combat, state, visual memory, and RGB come from the
neural model and its disclosed typed rules. The simulation target is 30 Hz;
`--fps 60` caps display updates and repeats the latest image when needed, rather
than adding learned intermediate frames. 128×128 is the native model resolution;
1024×1024 display uses classical resizing unless an optional trained SR model
is explicitly supplied. SR is outside the current experiment.

A standalone autonomous evaluation can be run after training:

```powershell
& $twPython -m turkeyworld.evaluate_rollout `
  --world training_runs/vjepa500_20261109/checkpoints/student_vjepa_temporal.pt `
  --device cuda --seconds 300 --scenario mixed --seed 41000 `
  --render-every 300 --min-render-psnr 35 --continue-on-failure `
  --output training_runs/vjepa500_20261109/manual_evaluation.json
```

One evaluation does not replace the pipeline's two independent checks.
Qualification examines integrity, meaningful movement, active NPCs, shots,
hits, kills, respawns, and image validity. Merely executing 9,000 ticks, surviving
while frozen, or displaying plausible images does not qualify. Oracle checks
measure errors without correcting the model. Existing seeds vary trajectories
on a fixed arena; generalization to new maps is not established.

## Verification

```powershell
& $twPython -m unittest discover -s tests -v
```

Dashboard worker/UI tests and its static Worker build use Node:

```powershell
Push-Location training_dashboard
node --test worker.test.mjs
node build.mjs
Pop-Location
```

Generated data, checkpoints, local logs, and `training_runs/` are excluded through
`.gitignore`. Save experiment artifacts separately when moving the project.
