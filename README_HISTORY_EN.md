# TurkeyWorld — change and results journal

[Русский](README_HISTORY_RU.md) | English | [Project instructions](README.md)

This separate journal covers the available chat history, from the first
prototype through RSSM, distillation and the paired comparison window. Repeated
requests to “launch it again” are grouped under the relevant mode. The sequence
of requests was reconstructed from the chat; numbers were checked against Git,
configurations and reports.

Results snapshot: **11 November 2026, 16:44:45 Moscow time, RSSM step 3450**.
Training continued after this snapshot. Times in this journal are Moscow time;
timestamps in the original JSON files are usually UTC. A run directory's name
does not always match the dates of every experiment inside it.

The following labels are used consistently:

- **Implemented** — the change exists in code or configuration.
- **Executed** — an artifact confirms the run, update or collection.
- **Quality passed** — the specified criterion was met on the specified panel.
- **Requested / planned** — an intention; completion has not been confirmed.
- **Accepted by the user** — the next stage was authorized while retaining the actual validation result.

Completing a budget, obtaining low training loss, matching a teacher and playing
correctly without corrections are different results. Metrics are comparable
only when their panels, targets, resolutions and loss definitions match.
Deleted old weights cannot be recovered from the remaining reports alone.

## H01. Prototype, conventional game and endless respawns

The game was launched first, followed by clarification about whether a trained
world model was already running. The first saved prototype used a Camera
Transformer + GRU and a direct RGB decoder; no claim was made that quality game
weights were available. Its sizes were Teacher 7 853 517, Student 2 067 341 and
SR 25 776 parameters. The historical state contained 90 values and eight NPCs.

Endless respawns were added on request. This did not mean infinitely many
simultaneous objects: the early limit was eight persistent slots. It was later
reduced to one. The conventional engine and the neural game received separate
launch commands and mode labels.

Evidence: first commit `a118abf`, [conventional game](turkeyworld/play.py),
[neural game](turkeyworld/neural_play.py).

## H02. Five minutes of play, Flow Matching, GAN and monitoring

The original goal was **300 seconds / 9000 transitions at 30 Hz**: prepare data,
train a teacher, then a student, and display progress on a web page. After a
question about the absence of Flow Matching, conditional Flow Matching was
chosen for the teacher and Rectified Flow for the student. Later requests added
a teacher of about 75M, adversarial rendering loss, images close to Source, and
a transition to the student when the teacher plateaued.

Flow/GAN, quality checks, stages and the dashboard were implemented. An early
saved rendering check reported SSIM 0.789511 and MAE 0.027560; the 75M version
reported SSIM 0.779579 and MAE 0.029799. The strict thresholds were not passed.
`.gitignore`, logs, checkpoints and Windows background execution were added,
with a system keep-awake guard that did not require the display to stay on.

Sources: [early Flow run](training_runs/flow_20261108/),
[ignore rules](.gitignore), [power guard](turkeyworld/power_guard.py).

## H03. Compact V-JEPA 2-AC and long autonomous checks

The user's selected option was implemented as a game adaptation of V-JEPA 2-AC
at roughly 125M, without official pretrained weights. The actual generator had
**125 439 429 parameters**. Requests included training on its own predictions,
persistent maps/object slots, combat and return episodes, and 9000 autonomous
transitions without engine corrections.

The best saved teacher report refers to global step **1967**: PSNR 21.6611,
SSIM 0.751088 and MAE 0.035417. Its local quality-step 1 already followed
pretraining and long-rollout training. The plateau handoff occurred at **4966**,
with quality still unqualified. Both 300-second checks completed, but neither
passed the playable criterion; the student reached repeated repair rounds.

The “What the model predicts / shows” preview was initially requested every
500 steps, then every 10. Merely completing a finite 9000-step run did not
establish controllable gameplay.

Sources: [125M migration](training_runs/flow_20261108/vjepa125_migration.json),
[best quality](training_runs/flow_20261108/teacher_quality_125m_best_report.json),
[handoff](training_runs/flow_20261108/teacher_handoff_vjepa125.json).

## H04. Growth to 250M and 500M; data up to five million

The teacher grew successively to **249 956 805** and **500 044 741** parameters,
transferring previous values and adding new blocks. Adam was recreated for
these migrations; this differs from resuming a compatible checkpoint without
resetting its optimizer.

The dataset request grew from roughly one million to five million examples.
The preserved preparation report states **157084 files**, collection completed
in **3883.63 seconds**, and a target of 5M RGB training transitions. The original
manifest was later deleted, so a complete exact recount is no longer possible.
Separately confirmed were 256 model-visited TRAIN clips and 8192 neural
transitions, subsequently labeled by Source without corrections fed back.

Sources: [250M](training_runs/vjepa250_20261109/),
[500M](training_runs/vjepa500_20261109/),
[remaining preparation report](training_runs/vjepa500_vqgan_20261109/status.json).

## H05. Student size and the first instructions

After discussing a student of about 20M, the target became **22.48M**. An early
plan listed **22 475 013 parameters**, but remained pending: it must not be
counted as completed distillation. A later native32 student baseline with a
shared codec used a different configuration. The user subsequently specified
separate Flow, Base and VQGAN sizes; see H18.

[English](README.md) and [Russian](README_RU.md) instructions were added,
covering collection, training, model structure, launching and result checks.

## H06. VQGAN from scratch and its use by the teacher

A VQGAN inspired by CompVis/taming-transformers was selected for training on
game frames. It was also incorporated into teacher rendering. The user stopped
codec training and requested continuation to the following stages.

The first codec had **4 763 731 parameters** and **11586 updates**; PSNR 31.1765,
SSIM 0.938776 and MAE 0.011958. Its automatic gate remained false. The teacher
with VQGAN had 504 857 752 parameters. In this run's branch, straight-through
gradient propagation through quantization was fixed while the forward result
remained bit-exact. A frozen codec did not automatically mean a trained renderer.

Sources: [first codec](training_runs/vqgan_20261109/codec_ready.json),
[its validation](training_runs/vqgan_20261109/codec_validation_final.json),
[500M + VQGAN](training_runs/vjepa500_vqgan_20261109/).

## H07. Loss noise, accumulation and renderer training

Discussion covered batch1, zigzagging loss, the distinction between training
and validation, and `condition`. To reduce noise, accumulation was requested
first over 24, then 8, and later 4 fragments. The preview interval was reduced
to 10. After the image generator was found to be frozen, the user explicitly
required renderer training too: subsequent plans included Base, Flow and GAN,
with separate codec-freezing and later-phase policies.

**Base** became the name of the direct RGB branch; Flow refines the image
through a latent representation. **Condition** means the states, actions, map,
memory, noise time and other explicitly specified features supplied to the
model, rather than the target future frame.

## H08. Determinism, a fresh start and a small overfit dataset

NPC spawning and movement were made deterministic; spawning was then simplified
to a game-time cycle. A fresh start for the world model and renderer was
requested while retaining VQGAN. The goal was shortened from five to **two
minutes / 3600 transitions**.

For the next 6000 updates, dataset sizes were discussed successively as
**400k →128k →32k**. The final intention was to fit the teacher to a small fixed
set first, then expand it. Updates sampled with replacement do not imply a full
pass through every unique example. The deterministic from-scratch run stopped
at 20; simple-spawn500M stopped at 542. These were intermediate stops, not
successful gameplay.

Sources: [from-scratch initialization](training_runs/vjepa500_deterministic_scratch_20261109/initialization.json),
[simple spawn](training_runs/vjepa500_simple_spawn_32k_20261109/).

## H09. Missing enemies: conditioning, masks and GAN

Source contained NPCs while the model drew walls and the weapon. Discussion
covered a weak GAN, frozen rendering, backgrounds, Base and the connection
between images and states. The code gained Source labels for visible NPCs,
extra weight on their pixels, images without NPCs as conditional negative
examples, and aligned GT-pose auxiliary objectives. Separate NPC/background
normalization was introduced later.

GAN alone did not establish correctly rendered objects or autonomous gameplay.
The absence of NPCs in the later navigation phase has a different explanation:
that training profile intentionally disables enemies.

## H10. Reduction to 175M, large batches and “Fragments B”

The teacher was reduced to about 175M, with everything except VQGAN restarted.
The actual early 175 profile had **174 910 616 parameters**. Accumulation16 and
successive microbatch increases were requested; the complete settings history
is listed below. The preserved simple-spawn175 run stopped at163.

The composition of the next dataset, **B**, was clarified as **64k total =16k
old +48k new**: the old portion was 25% of the new dataset, rather than 25% of
the previous32k. Preparing it did not automatically activate it for training.
The user later requested deletion of B and all previous checkpoints; that
dataset and the early `.pt` files are now absent.

Source: [175M fresh start](training_runs/vjepa175_simple_spawn_32k_20261109/initialization.json).

## H11. Native64 →native32 and fresh preparation

After training was stopped, requests specified native64, a trained SR×4 followed
by bilinear×4, retraining VQGAN, rebuilding128k and restarting
dynamics/Base/Flow/GAN. The target was then corrected to **native32** before
the actual fresh32 run. No preserved completed standalone native64 training
run was found.

The confirmed dataset had **4000 TRAIN clips ×32 =128000 transitions** and
**512 VAL clips ×32 =16384 transitions**. Each clip contains33 observations;
Source32 and genuine Source128 are aligned to the same states. This is128k
transitions, not128k files. SR and bilinear rows were added to the original
three-row preview. The new native32 baseline had **174 461 156 parameters**.

Source32→SR128→bilinear512 corresponds to two ×4 increases. Later game
windows used **1024**: their final128→1024 increase is ×8. The mode without
SR used direct bilinear enlargement from native32.

Sources: [128k manifest](training_runs/fresh32_128k_20261109/data/manifest.json),
[fresh pipeline](training_runs/fresh32_pipeline_20261109/).

## H12. Freezing VQGAN32 and improving the upscaler

The user fixed VQGAN32 at **42216 updates** and moved on to SR. The codec
retained a1488×16 codebook and an **8×8×16** latent; encoder+codebook+decoder
contain4 763 731 parameters. On128 held-out frames: PSNR36.1260,
SSIM0.965196 and MAE0.008895. **68/128 frames passed**, against a95%
requirement; the export records user acceptance rather than a passed gate.

The first SR32→128 completed2000 updates: SSIM0.698759 on100 frames.
After clarification of “90% similarity,” the target became **mean SSIM≥0.90**.
The stronger EDSR-inspired SR,1 517 571 parameters, completed20000 updates:
on the fixed512-frame panel, clean SSIM0.852550 and0.846338 after the codec.
The0.90 target was not reached. On that same512 panel, the earlier baseline
had0.705097/0.702811.

Sources: [codec32](training_runs/vqgan32_20261109/codec_ready.json),
[SR2000](training_runs/sr32x4_20261109/sr_validation.json),
[SR20000](training_runs/sr32x4_quality_20261109/sr_ready.json).

## H13. Spatial states, motion, combat and spawning

Player position, heading, map and the coordinates of all eight NPCs were linked
to prediction and rendering. Slots preserve ID, life state and HP outside the
camera. A stationary NPC's direction is not invented: direction is derived
from velocity with a validity flag. Local forward/strafe objectives, sliding,
stopping and returns were added.

Combat was split into target selection, attack, damage and death; actual hits,
misses and occlusions received additional emphasis. Nominal hit counts agree
with HP60→25→0 under35 damage; they are not a full shot history. Spawning gained
geometry, participant states, image auxiliaries and eight anchors with a
fallback. Coordinates, directions, HP and wall-cell/material tokens were
included in Base/Flow conditioning.

Spatial and combat migrations ran at steps66 and147. The generator grew to
175 585 830, then176 492 456 parameters. These counts include the codec;
EMA, critics and SR are counted separately.

Sources: [spatial migration](training_runs/vjepa175_32px_128k_20261109/spatial_entities_migration.json),
[combat migration](training_runs/vjepa175_32px_128k_20261109/combat_conditioning_migration.json).

## H14. Nine V5 improvements and a shorter duration

Added were own-trajectory XY objectives at1/4/8/16, circle/wall and X→Y path
penalties; positive/negative attack balancing; anchor+fallback spawning;
separate NPC/background reductions; Flow patch4→2 (**4→16 tokens**);
object/wall conditioning inside Flow blocks; decoded SSIM/edge terms;
noise-time stratification and diagnostics for five t ranges; and validation
every100. Focal loss remained optional and disabled by default.

V5 was enabled at725, with **181 167 857 generator parameters**. Eight patch
tensors with changed shapes were transferred deterministically and their Adam
states reset; compatible states were preserved. This introduced a new loss
protocol. The game target was shortened successively from **120→30→15
seconds**; the final target is **450 transitions**. Batch settings were
gradually reduced to16×1.

Source: [V5 migration](training_runs/vjepa175_32px_128k_20261109/training_v5_migration.json).

## H15. Four training scenes and one enemy

A curriculum was implemented: navigation without enemies →stationary NPC
→moving NPC with combat/death/spawning →full15-second game. The original
increase to eight enemies was canceled: the user limited the game to **one**.
Conditioning, heads, losses and data were rebuilt: **state90→20**, participant
tokens9→2. Source spawning uses a simple `(game_tick //90) %8` with a search
for a legal location, without hashing. One active life does not mean only one
respawn location.

The818→0 transfer preserved1561 compatible tensors and the codec, but created
Adam/D/EMA anew; the complete previous function was not preserved bit-exactly.
The new generator has180 071 125 parameters. Scene datasets were collected,
but navigation was not passed: `history=[]`, `final_qualified=false`.
The following scenes must not be counted as completed.

Sources: [one-enemy migration](training_runs/staged_single_20261110/single_enemy_migration.json),
[curriculum state](training_runs/staged_single_20261110/staged_curriculum_state.json).

## H16. Navigation sampling focus and separate component training

A sampling focus was enabled for sliding and returns: slide30%, return25%,
wall-stop20%, turn15%, straight5%, strafe5%. It began at625 and was superseded
at777 after152 updates /2432 draws, before completing its planned300 updates.

Three component phases were then implemented: **Base+Flow+GAN at GT poses**,
**Dynamics with the renderer frozen**, and **joint fine-tuning**. Renderer777→1444
was completed at the user's request after667 updates, with
SSIM0.703712/MAE0.031542 and a failed gate. Dynamics1444→1633 executed189
updates. The joint phase was implemented in the plan but never actually began.

Source: [component phases](training_runs/staged_single_20261110/component_training_state.json).

## H17. Long JEPA dynamics and numerical stability

Dynamics switched to450 own-state transitions from1520; the first confirmed
update was1521. The phase executed76 short and113 long450 updates. At batch16,
one long update contains7200 transitions and29 TBPTT16 chunks: numerical
coordinates/memory continue while the gradient graph is detached between chunks.

A nonfinite gradient before Adam1537 led to recovery from1530. The long path
was moved to FP32, gradient norms toFP64, and finite guards and saving after
every long update were added. Seven safe replay updates did not prove a single
root cause of the original error. Immediate H450 was specified for joint
training, but joint did not run before the architecture change.

Sources: [freeze verification](training_runs/staged_single_20261110/long450_freeze_verification.json),
[numerical recovery](training_runs/staged_single_20261110/numerical_recovery_control.json).

## H18. VQGAN →Base →RF distillation and one Euler step

The initial Flow≈3.5M limit was clarified as a limit on the Flow network itself,
with shared Base/codec separate. The order then became codec→Base→Flow, with
sizes of roughly1M/1M/**7M**. Training was executed; similarity to the teacher
is not the same as similarity to Source.

| Component | Parameters | Executed / selected step | Metric against its teacher | Gate |
|---|---:|---:|---|---|
| Compact codec | 997 539 | 6000 /5900 | decoder SSIM0.944053; encoder SSIM0.929427 | false |
| Base | 996 844 | 1100 /1100 | SSIM0.960514, MAE0.013893 | true |
| RF,4 Euler | 6 992 192 | 1000 /700 | SSIM0.908584, MAE0.017261 | false |
| RF,1 Euler | 6 992 192 | 1500 /1500 | SSIM0.911468, MAE0.017808 | false |

The1488×16 codebook was preserved. The selected compact assets were attached
for experimental preview while retaining the fact that the codec gate had not
passed. RF was genuinely trained for a one-step endpoint. The active
Base/RF/codec total is **8 986 575** parameters. Shared map/geometry and the
resident inactive Base must be accounted for separately. The first autonomous
RF1/JEPA450 test failed: final XY8.420354 tiles,284 wall steps; Source wall0.

Sources: [distillation](training_runs/renderer_distill_20261110T204351Z/),
[RF1](training_runs/renderer_distill_one_step_20261110T212351Z/).

## H19. Moving to DreamerV3-style RSSM25M

At the user's request, JEPA with its EMA encoder was replaced by RSSM: first
distill the previous dynamics, then train from Source. Its size is **25 007 048
parameters**, including the map and training-only observation decoder;
23 287 334 are trainable. Memory contains `h=1008`, `z=32×32`, totaling
**2032 FP32 values**. Runtime uses argmax; this adaptation has no EMA, actor or
critic.

Compatible render assets from teacher1633 were transferred; a new Adam was
created for the new dynamics. This is an RSSM adaptation, not the complete
DreamerV3 RL algorithm. Explicit states, the map and known heading/clock/event
rules remain part of the system. The first plan of **200 KD +1800 Source
=2000 updates** completed; the Source branch trains on450 own transitions.
A separate posterior stream observes Source for representation/KL without
replacing the main prior rollout.

Sources: [RSSM migration](training_runs/rssm25m_20261111/checkpoints/rssm_initial.pt.migration.json),
[runner](turkeyworld/train_rssm_dynamics.py). Corrupted RSSM labels in the UI
were also fixed.

## H20. Contact data, trials A/B and continuation after2000

Loss analysis found errors in blocking, sliding and long XY trajectories.
Trial A from339 was stopped at368 after held-out navigation regressed. Trial B
started again from saved339 with Adam/RNG; balanced physical motion, endpoint
margins and Source-relative wall loss were added without penalizing legal
Source clearances.

At2000, mean final XY over16 one-frame450 runs became0.483973 versus1.090515
at339; wall steps were987 versus1831. Quality remained false. At the user's
request, training continued without a finite budget, with validation **every250**,
preview10 and stopping after **two complete successful checks**.

For continuation2000, 32 TRAIN own trajectories and4864 short clips were
collected:3840 new Source +1024 exact replay. Source labels do not correct
runtime. A structural coverage-gate error was fixed:contact256 was supplemented
with32 genuine held-out idle windows; numerical thresholds were not relaxed.

A complete navigation check requires short velocity error≤0.35 tile/s,
false-stop and missed-block rates≤0.05; over450 steps, final XY mean≤0.10 tile,
p95≤0.25, zero wall penetrations/crossings and finite state/memory.
Stopping training requires two consecutive complete successful checks.

Sources: [contact plans](training_runs/rssm25m_contact_b_20261111/),
[continuation data](training_runs/rssm25m_contact_b_20261111/continuation_2000/contact_refresh_audit.json).

## H21. Parallel Base/RF improvement with the codec frozen

A separate renderer refinement run at genuine Source poses completed2000
updates; **refinement100** weights were selected, not the final2000. RSSM339,
map/geometry, codec, appearance and SR remained bit-exact. The optimizer updated
only Base/RF1.

On128 held-out Source frames, Base SSIM improved0.4082→0.6578 and RF1
0.5775→0.6603; selected RF1 MAE was0.047796. The default TRAIN-pose RF1 result
was0.7701. The Source goal of SSIM≥0.90 and MAE≤0.03 was not passed. The active
renderer pointer is separate from the frozen copy inside the dynamics worker.

Sources: [selected validation](training_runs/renderer_refine_rssm25m_20261111/validation_selected.json),
[active renderer](training_runs/active_renderer.json), dashboard8768.

## H22. Loss, the actual neural game and video

Mean training losses over250-update windows2001–3000 were
**2.927→2.754→2.541→2.484**. Roughly1.1 of the loss is the configured KL floor;
the main reducible component is autonomous-trajectory coordinates. Low loss did
not prevent long-run errors. At3000, short-step velocity improved by about a
third relative to2000, but16 one-frame450 runs produced final XY mean0.555281,
p952.205283 and837 wall steps out of7200. No overall quality gate was passed.

The separate full-game-path check at3000 with FP16/RF1/SR1024 gave final drifts
of0.198125 for ordinary
movement,0.195448 for return and2.203878 for sliding; the latter entered a wall
at step49 and had38 endpoint wall steps. These are three scenarios, not the same
16-episode panel.

A [step2000 video](training_runs/neural_video_20261111T072305Z/capture/neural_game.mp4)
was recorded:450 transitions /15 game seconds, around32 seconds of actual
footage, H2641024×1024. The only Source RGB was the initial image; there were
no later corrections. The recorded speed includes video capture and is not an
isolated FPS benchmark.

Sources: [3000 analysis](training_runs/rssm25m_contact_b_20261111/manual_validation/analysis_20261111_g3000/),
[runtime report](training_runs/rssm25m_contact_b_20261111/manual_validation/analysis_20261111_g3000/runtime_validation/report.json).

## H23. Four V3 navigation improvements

The audit found that the new paired dataset had **0/384** starts near two
close walls: the free-tangent filter rejected corners. The main `context2`
branch also skipped transition0→1 even with one observed frame, while sample
mode always coincided with one dataset and one context.

All four requested changes were enabled from **3151**:

1. A new TRAIN pool of **2048 corner +4864 exact replay =6912 clips /
   221184 transitions**.256 distinct poses,4 quadrants; VAL exclusions≥0.25
   tiles were retained. All400 previous held-out files remained unchanged.
2. Half the450 runs use genuine `state0/action0/context1`; the others use two
   context frames. The window is explicitly selected from the beginning.
3. Up to4 own H16 branches take the actual detached state **and h/z** before
   an incorrect contact. Source supplies only separate targets; there is no
   new RGB, memory replacement or correction to the main trajectory.
4. A16-update cycle covers every pool×context with3argmax:1sample; best weights
   are saved separately using both450 checks: wall count first, then p95 and
   mean XY. A single successful check does not constitute qualification.

The actual run exposed a mining error: progress0.7 on a free axis incorrectly
excluded later contacts. The physical threshold, FB/BF/BB priority and retention
of pre-wall anchors were fixed; training resumed from3160 without rolling back9
updates. Model/Adam/RNG were checked bit-exact on resume, as were1156 frozen
tensors3151→3160. This snapshot passed **1035 Python tests and45 UI tests**.

The **3250 check failed**: ctx1 final mean0.705833, p952.140181 and919/7200 wall
steps; at3151 the figures were0.472222,1.573206 and883/7200. Best remains3151
with `qualified:false`; no V3 quality improvement was claimed.

Sources: [corner audit](training_runs/rssm25m_contact_b_20261111/navigation_revision/corner_dataset_audit.json),
[resume](training_runs/rssm25m_contact_b_20261111/navigation_revision/resume_verification_g3160.json),
[3250 validation](training_runs/rssm25m_contact_b_20261111/validation/step_003250/motion_recovery_validation.json),
[best selector](training_runs/rssm25m_contact_b_20261111/navigation_best.json).

## H24. One window with the conventional and neural games

A paired window was launched on snapshot **3420**: Source on the left, the
neural game on the right, sharing keys and one fixed dt per step. The
conventional side waits for the neural side; autonomous model coordinates are
not copied from Source. Tab enables a separately labeled renderer comparison
using a shared Source camera, without changing the model's own h/z. Enter
starts, P pauses, R resets both sides, and Esc closes the window. At450 steps,
the window stops and remains open.

There were10 CPU tests and a separate4-step GPU smoke check; this is not a new
combined full-suite count. The final user trial completed450/450 shared steps
after4 resets, with final XY0.057327 tiles, finite final state/h and no Source
correction. This is one manual trial, not held-out qualification. The window
closed on **11 November16:44:06 Moscow time**; its launch pointer could remain
stale.

Sources: [paired runtime](turkeyworld/compare_runtime.py),
[paired GUI](turkeyworld/compare_play.py),
[session result](training_runs/paired_compare_20261111T133512Z_g3420/report.json).

## H25. This journal and the next renderer task

Separate Russian and English histories of all changes and results were
requested. These two READMEs were added with matching stages and evidence
links. An addition after the main snapshot: a separate training run,
`renderer_enemy_refine_20261111`, started on **11 November17:20:48 Moscow time**,
with dashboard **http://127.0.0.1:8769/**. Navigation-RSSM continued training
independently.

The compact **Base996844 and RF6992192** were updated, using one Euler step;
codec997539, RSSM, map/geometry, appearance and SR stayed frozen. Training used
genuine one-NPC Source frames and GT poses, exact visible masks and backgrounds
without NPCs. In batch8,75% were visible NPC examples and25% were negatives,
split equally between no NPC and a living NPC outside the camera/behind a
wall. Static snapshots use zero appearance features; dynamic examples use
only the frozen prior's previous RGB/state/action. This trains rendering at
specified states, not combat dynamics.

The first **2000-update series completed**; validation100, preview10. The fixed
256-frame check contains192 visible NPCs,32 no-NPC examples and32 with a living
but invisible NPC.133 target frames were excluded from VAL because their scenes
matched TRAIN; the source data was not rewritten. Static/dynamic domains,
large/partially occluded/distant enemies, corpses and both negative categories
were checked separately.

**Step100** was selected: RF foreground MAE0.163369→0.158536,
Base0.160659→0.150672. The fraction of RF frames closer to the Source NPC
than to the Source background without an NPC was0.03125. This is a diagnostic
indicator, not a detector or a gameplay success percentage. At2000, RF
foreground MAE was0.15920; reliable NPC rendering was not learned, the absolute
gate is false, and the active renderer was not replaced. All frozen
parameters/buffers remained bit-exact.

The frozen compact codec's reconstruction reference had NPC MAE0.133639;
this is not a proven limit on the generator's attainable quality. After the
first series, a separate bounded diagnostic of conditioning, gradients and
Original-E/Q→compact-D began on TRAIN examples while VQGAN remained frozen.
No result is claimed in advance for subsequent training.

Sources: [first enemy run](training_runs/renderer_enemy_refine_20261111/),
[selected100 validation](training_runs/renderer_enemy_refine_20261111/validation/step_000100.json),
[data indexing](training_runs/renderer_enemy_refine_20261111/data_audit.json).

## Changes to goals, batches and data units

| What changed | Chat sequence / outcome |
|---|---|
| Game duration | 300→120→30→15 seconds;9000→3600→900→450 transitions at30Hz |
| Teacher size | 75M→125M→250M→500M→175M; spatial/V5 subsequently changed the actual total |
| Dynamics | Camera Transformer+GRU →compactV-JEPA2-AC+EMA →RSSM25M |
| Training data | about1M→5M; fixed overfit400k→128k→32k; B64k; then a new native32 corpus of128k **transitions** |
| Native and display | request64→32;SR×4;bilinear512 through two×4 increases; later game windows1024 |
| Simultaneous NPCs | 8→1; endless sequential respawns retained |
| Image previews | 500→10 updates;3→5 rows;H4 is not a15-second video |
| Validation | 100 in the old world/component phases;250 in current RSSM continuation |

The requested sequence of **microbatch ×accumulation =effective batch**:

```text
1×24=24 →1×8=8 →1×4=4 →1×16=16 →3×16=48
→10×16=160 →24×16=384 →48×16=768 →48×21=1008
→21×21=441 →21×5=105 →32×1=32 →27×1=27 →16×1=16
```

This is a request history, not equally complete telemetry for every early run.
Later safe-resume evidence confirms746→747 for27×1 and800→801 for16×1.
Current RSSM uses main-batch16, accumulation1; separate Source auxiliary clips
and up to4 contact branches are counted separately.6000 Adam updates do not
equal6000 examples or epochs. Draws can select the same clip repeatedly.

## What remained unconfirmed at snapshot3450

- Stable autonomous15-second navigation has not passed the complete held-out panel.
- The full NPC/combat game curriculum after navigation has not completed.
- Source rendering quality≥0.90 and SR≥0.90 were not reached in the specified checks.
- There is no winning `best_qualified`. Best physical weights differ from the latest Adam checkpoint.
- Snapshot3450: training loss1.915361, without the KL floor0.815361, contact-branch loss0.026969;
  this is one training row, not new validation or proof of quality.
- Early checkpoints and dataset B were deleted. A JSON file or a path inside it does not establish that a `.pt` file exists.

## Git checkpoints

Commits reflect code state, not training success. Later changes may remain in
the working tree.

| Moscow time | Commit | Subject |
|---|---|---|
| 08.11 22:00 | `a118abf` | first step |
| 09.11 10:12 | `cc2df25` | jepa and rectflow in action |
| 09.11 13:10 | `09ccddd` | +grad accum |
| 09.11 19:44 | `cbe986b` | 175m params |
| 09.11 22:53 | `6b80964` | game size32×32 |
| 10.11 01:56 | `6885a44` | spatial /fight /move loss |
| 10.11 12:03 | `88e2f6b` | improvements loss |
| 10.11 13:00 | `586e8cb` | curriculum learning |
| 10.11 19:36 | `f042c64` | renderer →dynamics →joint plan |
| 11.11 00:41 | `69a086c` | distil render |
| 11.11 09:58 | `25be7a4` | JEPA →DreamerV3 |

## Continuing this journal

Add each new H stage to both language versions, preserving earlier results.
Each entry should state the request, change, steps actually executed,
protocol/panel, metric value, gate, artifact path and weight availability.
Current `status.json` changes; numerical conclusions should be pinned to a
specific step and report. Record budget completion, user acceptance and quality
passes as separate events. The main instructions remain in the READMEs.

Most local `training_runs/` are excluded from Git: their links work in this
workspace and do not promise data/weight availability in a clean clone.
