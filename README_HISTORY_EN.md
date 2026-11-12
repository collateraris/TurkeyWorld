# TurkeyWorld — change and results journal

[Русский](README_HISTORY_RU.md) | English | [Project instructions](README.md)

This separate journal covers the available chat history, from the first
prototype through RSSM, distillation and the paired comparison window. Repeated
requests to “launch it again” are grouped under the relevant mode. The sequence
of requests was reconstructed from the chat; numbers were checked against Git,
configurations and reports.

Original results snapshot: **11 November 2026, 16:44:45 Moscow time, RSSM step 3450**.
Later H entries record subsequent runs, including the continuation from5112 in H53.
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
steps; at3151 the figures were0.472222, 1.573206 and883/7200. Best remains3151
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

## H26. NPC image diagnosis and correction after the frozen codec

After the first H25 series, a separate test used **16 large NPC examples from TRAIN**.
This is a diagnosis, not held-out validation. Reconstruction foreground
MAE was: compact-E→compact-D **0.134384**, original-E→compact-D **0.135389**,
original-E→original-D **0.061319**. Replacing only the encoder did not help.
Changing the alive feature changed RF RGB by **0.011637** on average, and moving
the NPC by **0.003611**: conditioning is connected, but this does not establish
correct rendering. On these same frames, gradient norms were Base **0.486453**,
RF **0.207275**; suppression by global clipping was not confirmed here. A private
Base TRAIN test over **64 steps** reduced ROI MAE **0.152507→0.125380**, but the
diagnostic NPC presence fraction stayed **0%**. All state in both codecs stayed bit-exact.

An optional neural RGB correction was added for the next experiment after the frozen
decoder. It uses current Base query features from state/map/appearance, with
no Source RGB at inference. Zero initialization preserves the previous image;
old bundles leave this branch disabled by default. The additional Linear
**144→12** has **1740 parameters**: Base **996844→998584**, RF stays
**6992192**, and the codec **997539**.
Formula: `clamp(decoded_rgb + 0.5*tanh(residual), 0, 1)`.
Neither VQGAN parameters nor its buffers are trained.

The implementation passed **43 focused CPU tests**: old and new bundles, zero
initialization, export/load and actual NeuralRuntime. Training this branch
was not launched: the next request changed the priority to fine-tuning the
compact VQGAN. The codec freeze applies to this residual path; separate
compact codec fine-tuning is now explicitly authorized by the user.

Sources: [TRAIN diagnosis](training_runs/renderer_enemy_refine_20261111/diagnostics/renderability_probe.json),
[image comparison](training_runs/renderer_enemy_refine_20261111/diagnostics/renderability_probe.png).

## H27. Fine-tuning the compact VQGAN on the enemy

The next request was to fine-tune the compact VQGAN on enemies.
It explicitly authorizes compact encoder/decoder updates, overriding the
previous freeze for this separate experiment. The original teacher VQGAN and
the current RSSM are unchanged. Its size stays at **997539 parameters**; the codebook,
all quantizer buffers and latent normalization stay fixed. The new codec
does not automatically replace the active renderer.

The initial budget is **2000 updates**, batch **8**, validation **100**,
preview **10**. The first **200** updates are decoder-only with
`originalE(Source)→Q→compactD→Source`; the next **1800** train encoder+decoder.
Learning rate: encoder **0.00002**, decoder **0.0001**.
Foreground/background errors are normalized separately by area; SSIM,
edges and contrast are included. Replay of actual old RF endpoints and
encoder alignment on Base RGB are retained. GAN is disabled in this series:
this fine-tunes an existing VQGAN's E/D against Source and frozen references.

Reconstruction, decoding previous RF endpoints and the full
`RF(newE(Base))→newD` need separate checks: a fixed codebook and scale alone
do not guarantee compatibility. The **256**-frame held-out panel is
retained. The objective and **4 CPU tests** are ready; the runner and codec
replacement path were also checked with separate CPU tests.

**Executed:** `codec_enemy_refine_20261111` started on **11 November
18:07:02 Moscow time**, worker **31132**, dashboard **http://127.0.0.1:8770/**.
The first **200** D-only updates completed; E/D weights are now actually updating.
At snapshot **300**, Source reconstruction NPC ROI MAE was **0.133639→0.116566**,
and the diagnostic fraction closer to Source-NPC than background **0.088542→0.489583**.
This is not an enemy detector or a gameplay success measure.
For unchanged RF weights with new E/D, foreground MAE was **0.158536→0.156655**;
this is not yet reliable neural NPC rendering. Empty/RF regression guards passed,
but the combined Source quality gate is false. All Q parameters/buffers and the
original world/Base/RF/map/original-codec stayed bit-exact. Training continued
after this point; the historical **300** snapshot is not the current live step.

**The series completed at2000, selecting2000.** Source E+Q+D foreground MAE
was **0.093483**, SSIM **0.879107**, and full-frame MAE **0.022894**.
Foreground MAE decreased by approximately **30%** from its initial value;
the diagnostic NPC fraction was **0.838542**. Original-E→new-D foreground MAE
was **0.092033**. Empty/RF guards are true; the full Source gate is false.
Original RF weights with new E/D give foreground MAE **0.157057** and SSIM
**0.617499**: this confirms that separate renderer adaptation is still needed.
Base code agreement between the old and new encoder is **0.616943**; a fixed
codebook does not imply unchanged encoder outputs.
The joint phase actually trained **973731** E/D parameters; **23808** Q
parameters and all its buffers stayed unchanged. The selected codec was
exported separately; the active game was not switched automatically.

Sources: [objective](turkeyworld/codec_enemy_objectives.py),
[update and freeze checks](tests/test_codec_enemy_objectives.py),
[codec run](training_runs/codec_enemy_refine_20261111/),
[validation300](training_runs/codec_enemy_refine_20261111/validation/step_000300.json),
[final validation](training_runs/codec_enemy_refine_20261111/validation_selected.json),
[compact codec](training_runs/codec_enemy_refine_20261111/checkpoints/codec_student.pt).

## H28. Base/RF adaptation with the fine-tuned frozen codec

A separate `renderer_enemy_codec_adapt_20261111` started on
**11 November18:20:37 Moscow time**, worker **3532**, dashboard
**http://127.0.0.1:8771/**. The new codec was selected at2000 from H27 and is
fully frozen; Base **996844** and RF **6992192** continue training with one Euler step.
Initial RF/Base weights, Q, normalization and fixed noise were checked bit-exact;
export and strict reload of the new bundle passed. The original source,
active pointer and parallel RSSM were unchanged.

New Adam, budget **2000**, batch **8**, validation **100**, preview **10**.
Alongside pixel/SSIM/edge/contrast, opt-in balanced endpoint latent MSE was added
with weight **0.1** against `frozen compactE+Q(Source)` in the original normalization.
Source RGB is only a loss label. Both the codec and objective changed, so
metric changes cannot be attributed to a single factor.
Default weight **0** preserves the previous loss; H26 residual and GAN are disabled.

The first held-out snapshot **100**: RF foreground MAE **0.157057→0.147811**,
diagnostic NPC fraction **0.036458→0.119792**, Base **0.150672→0.149256**.
There is `rf1_empty_navigation_regression`; experimental selection and the full
Source gate are false. All frozen parameters/buffers, including the new codec,
stayed bit-exact. This confirms actual training began, not a finished game;
training continues in the background.

At **700**, RF foreground MAE is **0.138225** and the diagnostic NPC fraction
**0.270833**. Relative experimental selection is now true:
this check has no empty-scene regression. The full Source gate and NPC
appearance target remain false; frozen-component invariance was confirmed.

Sources: [new renderer run](training_runs/renderer_enemy_codec_adapt_20261111/),
[codec replacement proof](training_runs/renderer_enemy_codec_adapt_20261111/codec_swap_proof.json),
[check100](training_runs/renderer_enemy_codec_adapt_20261111/validation/step_000100.json),
[check700](training_runs/renderer_enemy_codec_adapt_20261111/validation/step_000700.json).

## H29. Paired Source and neural-renderer window with a shared camera

The `compare_play` window actually launched on **11 November18:41 Moscow time**,
PID **14692**, session `paired_compare_20261111T153932Z_enemy_codec`. It uses
RSSM snapshot **3780**, renderer selected at **1900** from H28, Base **996844**,
RF **6992192**, the new **997539** codec (`67bfff93…`), one Euler step and SR
from the20000-step series. The window is **1572×968**, with two768-pixel panels;
Source renders at256, while the neural chain is **32→trained SR128→bilinear1024**,
then fitted into its panel. One keyboard action is applied to both sides per
paired tick; simulation30Hz, trial450 steps, catch-up limited to one step.

The initial **`shared_camera`** mode uses the Source pose only to display the
neural renderer from the matching camera. It does **not** correct the RSSM's
own state or memory: `engine_corrections=0`, `rssm_future_source_feedback=false`.
**Tab** switches the display to autonomous mode. The profile is
`navigation_only=true`; this launch does not validate full combat or qualify
the game. Initial readiness was paused at0; the saved `ready_snapshot.json`
already records user play at **39/450**, unpaused, with finite state/memory.
The current `status.json` changes. Renderer1900 passed relative-improvement
and empty-scene checks, but its full Source gate remains false
(RF SSIM **0.691841**, NPC ROI MAE **0.134537**, diagnostic NPC fraction **0.302083**).
A shared-camera comparison does not establish autonomous navigation quality.

Sources: [command and selected weights](training_runs/paired_compare_20261111T153932Z_enemy_codec/launch.json),
[saved readiness at39/450](training_runs/paired_compare_20261111T153932Z_enemy_codec/ready_snapshot.json),
[current window status](training_runs/paired_compare_20261111T153932Z_enemy_codec/status.json),
[selected renderer1900 check](training_runs/renderer_enemy_codec_adapt_20261111/validation_selected.json).

## H30. New Source corpus: one enemy, combat and detailed navigation

On **11 November**, a separate immutable corpus was collected in
`training_runs/combat_source_20261111/data`: **1280 NPZ files**, native32, state20,
one NPC,30Hz, policy `deterministic_enemy_v4_single_spawn_cycle`.
Previous datasets and fixed validation panels were not overwritten.

| Subset | TRAIN: files / actual transitions | VAL: files / actual transitions |
|---|---|---|
| moving1, short combat | 512 /15362 | 128 /3844 |
| full_game,451 transitions | 256 /115456 | 64 /28864 |
| navigation, no NPC | 256 /8192 | 64 /2048 |
| Total | 1024 /139010 | 256 /34756 |

Combat includes duel, melee_then_shoot, miss_then_hit, blocked, kill_respawn and
return. The short subset also retains genuine terminal episodes:
**73 TRAIN and18 VAL**, ending before32 transitions. They are not padded with
fictional survival and cannot supply a450-step target. All **256/64** long
TRAIN/VAL episodes support horizon450 with contexts1/2. Navigation includes
corner, slide, return, wall_stop, turn and strafe; NPCs are explicitly disabled.

NPZ files contain actual Source RGB, states, actions, events/hits/attacks,
exact visible-NPC masks and Source RGB without the NPC, plus depth/visibility.
An independent audit verified SHA256 for all **1280** files, no overlap in seeds,
complete NPZ files or exact initial states between TRAIN/VAL, empty navigation
NPC masks and genuine corner geometry. One episode from each case was replayed
in Source: states/events matched bit for bit, and selected RGB frames were also
checked. This establishes episode/seed separation, rather than claiming every
VAL camera pose is novel.

Sources: [manifest](training_runs/combat_source_20261111/data/manifest.json),
[independent audit](training_runs/combat_source_20261111/data_audit_independent.json),
[exact collection command](training_runs/combat_source_20261111/launch.json).

## H31. Combat RSSM and parallel Base/RF training on new Source frames

The next user request split training into two independent processes.
The H29 game window is not restarted, and new weights are not automatically
attached to it.

**RSSM:** `training_runs/rssm25m_combat_20261111`, dashboard
**http://127.0.0.1:8765/**. The navigation run stopped at **3873**;
migration preserved all **1271 tensors**, the existing **115 Adam parameter names**,
**102 existing optimizer states**, RNG and history. The only semantic
configuration change is `navigation_only:true→false`.
The first actual update **3874 was saved**: combat heads acquired Adam states
for the first time, bringing the total to115; this does not reset existing
moments or add model parameters. Base/Flow/codec/map/appearance were confirmed
bit-identical after that update.

The cycle contains **3 combat updates and1 explicit no-enemy navigation replay**,
batch16×accumulation1,450 own transitions and memory, TBPTT16.
Supervision adds shot targets, hits/misses, attacks, NPC/player damage,
NPC death, NPC motion and spawn. Auxiliary Source labels do not replace the
independent prior with future RGB or true state. Validation every250 updates
combines the previous navigation criteria with separate held-out heads/16×450
using one NPC; previews run every10. There is no fixed budget: two consecutive
complete checks must pass. Baseline3873 **failed both navigation and combat**;
gameplay qualification is not claimed.

Strict Base/RF binding previously included the navigation-only profile. A new
helper first validates the old bundle against its original lineage, permits
only this flag change with every world tensor identical, then creates a
separate combat-profile bundle. Base/RF/Q/normalization/noise and previous
quality findings remain unchanged; original artifacts are not overwritten.

**Renderer:** `training_runs/renderer_combat_navigation_20261111`, separate dashboard
**http://127.0.0.1:8772/**. Initial weights are Source RSSM snapshot **3780** and
renderer **1900** from H28. Base **996844** and RF **6992192** train with one Euler
step; VQGAN **997539** (`67bfff93…`), RSSM, map, geometry, appearance and SR stay
frozen. GAN and the optional pixel residual are disabled. The new dataset
explicitly starts a fresh Adam. Each batch16 contains
**8 combat +4 appearance +4 detailed-navigation** frames.
Combat uses actual hit/miss/player_hurt/death/respawn labels; navigation uses
corner/slide/return/wall_stop/turn/strafe. Source supplies fixed poses, while
dynamic features use only the previous observation. The Source latent target
has weight **0.1** and is a loss label only.

After excluding **328** VAL frames matching exact TRAIN scenes, there are **146386 TRAIN** and
**31277 VAL** target frames; the fixed panel has **256** frames. Sampling quotas,
RNG and pool/cohort counters are checkpointed. Validation runs every100 updates,
preview/save every10. At2000, update2000 was selected: Base SSIM **0.706131**,
MAE **0.040099**, NPC ROI MAE **0.121004**; RF SSIM **0.709609**, MAE **0.041431**,
NPC ROI MAE **0.117572**. Relative-improvement checks passed; the complete Source
gate remained **false**. These are fixed Source poses, not autonomous combat.

The same run therefore continued **2000→total6000**, changing only the budget.
The saved2100 checkpoint confirmed all **317 Adam states** at step2100, the same
frozen fingerprint and Torch RNG;33600 draws split **16800/8400/8400**.
New results do not automatically replace the active renderer. Navigation,
combat, Source image quality and gameplay qualification remain separate results.

Sources: [RSSM command](training_runs/rssm25m_combat_20261111/launch_command.json),
[migration](training_runs/rssm25m_combat_20261111/combat_migration.json),
[renderer binding](training_runs/rssm25m_combat_20261111/renderer_profile_rebind.json),
[baseline3873](training_runs/rssm25m_combat_20261111/validation/step_003873/rssm_combat_validation.json),
[renderer command](training_runs/renderer_combat_navigation_20261111/launch_command.json),
[sampling audit](training_runs/renderer_combat_navigation_20261111/data_preflight.json),
[Source check2000](training_runs/renderer_combat_navigation_20261111/validation_selected_2000.json),
[Adam/RNG continuation proof](training_runs/renderer_combat_navigation_20261111/continuation_2000_to_6000.json).

## H32. Numeric NPC feedback gradient amplification and revision V2

The first **6 combat updates3874–3879** were finite, but gradient norms reached
roughly **1e20–1e23**. The run stopped safely at3879, preserving all six updates;
there was no rollback. Finite values alone did not mean the training scale
was healthy.

An isolated probe on the same TRAIN episodes, checkpoint3879, batch16 and H450
compared gradient paths without an optimizer step:

| Variant | Main gradient norm |
|---|---:|
| Previous numeric feedback, chunk16 | 2.219e23 |
| Same feedback, chunk4 | 2.312e5 |
| No gradient through NPC velocity feedback | 7.124e6 |
| No gradient through all numeric state feedback, chunk16 | 165.80 |

The separate auxiliary norm was **1209.56**. All450 output states matched bit
for bit; memory was directly compared at offsets **1/16/450**.
This measures one fixed batch, rather than guaranteeing future stability or
establishing gameplay quality.

Immutable `numeric_feedback_revision.json` takes effect at3879 with protocol
`rssm_one_enemy_combat_numeric_feedback_v2`. Only on combat updates, the numerical
own state is detached from autograd before the next transition. Its values are
not reset or replaced with Source; **h/z BPTT16 is preserved**, as are450
transitions. The architecture, pre-resume weights, LR, dataset and thresholds
did not change. The command uses `--feedback-revision-plan`; the base plan is unchanged.

Restoration at3879 verified model, Adam and all RNG fields bit for bit — **1618
compared tensors**. The first saved V2 update **3880** confirmed **1156 frozen
tensors** unchanged, **115 trainable tensors** changed, the same optimizer
names/groups and115 optimizer states. Measured norms were:3880 combat **1279.73**,
3881 navigation **10.12**,3882 combat **1603.61**. Training continued, but these initial
checks do not establish navigation, combat or Source image quality.
The game window and independent renderer worker were not restarted by this revision.

Sources: [isolated probe](training_runs/rssm25m_combat_20261111/diagnostics/combat_gradient_probe.json),
[V2 plan](training_runs/rssm25m_combat_20261111/numeric_feedback_revision.json),
[restoration without rollback](training_runs/rssm25m_combat_20261111/resume_verification_g3879_detached.json),
[first V2 update](training_runs/rssm25m_combat_20261111/first_numeric_feedback_update_verification.json),
[current command](training_runs/rssm25m_combat_20261111/launch_command.json).

## H33. HP0 codec, controlled RF pilots and continued renderer training

The renderer loss audit at6000 showed a real but slow decline: the first1000
updates averaged **0.35057**, the last1000 **0.28410**; the last two500-update
windows were **0.28584→0.28237**, with individual-batch deviation around0.0223.
Held-out RF/NPC metrics were fluctuating. RF often worsened Base, especially
on player damage and large NPCs. Renderer gradients were neither absent nor exploding.

On the exact256-frame panel, the frozen codec had **10 HP0 frames**: Source
reconstruction MAE **0.123816**, SSIM **0.569506**, versus MAE **0.024230** for
the remaining246. A separate genuine HP0 panel has **18** frames;
10 overlap the main panel, giving264 distinct frames. Source-visible render
conditions separate it from TRAIN; the main fingerprint `8f736fc8…` is unchanged.
This measures a death/red-tint reconstruction problem, but does not establish
a mathematical codec ceiling or absence of such examples from all prior training.

The separate `codec_battle_terminal_20261111` run trained compact E/D only,
**997539 parameters**; the entire1488×16 quantizer, its buffers and latent
normalization were preserved. After2000 updates,1900 was selected: HP0 MAE
**0.015090**, SSIM **0.969407**; alive MAE **0.021686**, NPC ROI MAE **0.079946**.
Relative alive/NPC/empty/full-RF guards passed; the complete Source gate is still
**false**. These are learned weights, not a runtime tint effect.

Three sequential200-update pilots then started from the same renderer5800
Base/RF weights and Source RSSM snapshot3780. Every pilot pinned **the same
codec1000**, tensor SHA `ea1f1556…`; final1900 was not substituted between arms.
All200 training batch indices, initial predictions, RNG and256+18 panels matched.
GAN/residual/runtime blending were disabled.

| Pilot | RF→Base gradient | latent / guard | RF SSIM / MAE | NPC MAE | HP0 RF MAE |
|---|---|---|---|---|---|
| A, previous objective | no | 0.1 /0 | 0.740434 /0.035636 | 0.104265 | 0.032907 |
| B, coupled package | yes | 0.03 /1 | 0.736241 /0.036002 | 0.103531 | 0.033127 |
| C, guard only | no | 0.1 /1 | 0.741964 /0.035432 | 0.102958 | 0.032084 |

B was rejected: Base MAE was **0.036144**, versus **0.034401** for A; HP0 Base
was **0.037958**, versus **0.032258**. B's lower raw loss did not demonstrate an
improvement: its latent weight was smaller. The common RGB and old-latent0.1
objectives used identical weights; A/C RGB objectives were nearly equal at
**0.232086/0.232159**, versus B's **0.236793**. C's small Source RF/NPC/HP0
advantage does not establish improvement across every metric or seed.
**All three release gates were false**.

The guard is a separate per-image FG/BG penalty when RF worsens detached Base.
It cannot guarantee new-frame results. Release additionally requires RF no worse
than current Base and Base no worse than **original parent5800**, across overall,
alive/hurt/HP0/corpse/empty/navigation/large cohorts: MAE/region tolerance0.002,
SSIM0.01. Source and reference receive no gradient. Previous detach/loss defaults
are preserved; architecture and parameter counts did not change.

C200 was selected for further training in `renderer_repair_corrected_20261111`,
dashboard **http://127.0.0.1:8774/**. This explicitly attaches final **codec1900**,
tensor SHA `e7c82c5e…`, and freezes it again. C200 Base/RF tensors remain identical;
new Adam and sampler seed173 are declared as a separate phase with a6000-update
control budget. The original Base anchor is not reset to C200. Saved local620
confirmed317 Adam states; frozen check600 preserved all codec/world/map/appearance
tensors. **Full release has not passed**; game weights and the window were not
automatically switched.

Sources: [loss/HP0 audit](training_runs/renderer_combat_navigation_20261111/diagnostics/loss_analysis_6000.json),
[selected codec](training_runs/codec_battle_terminal_20261111/validation_selected.json),
[A/B/C and identical-selection proof](training_runs/renderer_repair_pilots_20261111/pilot_comparison.json),
[shared pilot initialization](training_runs/renderer_repair_pilots_20261111/initialization_proof.json),
[explicit new codec phase](training_runs/renderer_repair_corrected_20261111/initialization_proof.json),
[continuation plan](training_runs/renderer_repair_corrected_20261111/phase_plan.json),
[saved working checkpoint check](training_runs/renderer_repair_corrected_20261111/running_verification.json).

## H34. Combat SR x4 continuation and the requested2000-to10000 extension

The user requested SR training alongside dynamics and the neural renderer.
The existing **1,517,571-parameter EDSR** (64 channels,16 blocks, x4) was loaded
from its trained20000-step checkpoint. Its architecture and parent weights on
disk were preserved; this was a separate SR-only optimizer and run.

`sr_combat_data` replayed immutable Source initial states and recorded controls.
Every selected native32 image matched the stored Source RGB bit for bit;
**HR128 was rendered at exactly the same physical state**, never interpolated.
The cache contains **3072 TRAIN and384 VAL target pairs**:25% navigation
(768/96), with the remainder balanced across hit, miss, melee, NPC death,
respawn, wall-blocked melee and genuine player HP0. All73 TRAIN and18 VAL HP0
targets were retained. Each file also stores its preceding observation, so
6144/768 stored images are not6144/768 independent target examples. The SR
sampler uses only target frame1. Source TRAIN/VAL seeds do not overlap.

Training used batch8, FP32, AdamW LR`5e-5`, and actual-HR-foreground/background
balanced L1 +0.2 SSIM +0.1 edge loss. Approximately25% of training inputs were
reconstructed by the already trained compact HP0 codec; the other inputs were
clean Source32. The complete codec, including codebook and usage buffers,
remained frozen. Generated world/RF images were not paired with invented HR.
World dynamics, Base/RF and the active game SR were not changed by this run.

The first **2000 additional updates completed**, and selected update2000 was
exported at **22000 cumulative SR updates**. Fixed episode-held-out results:

| Input / cohort | SSIM before → selected2000 | MAE before → selected2000 |
|---|---:|---:|
| Clean Source32,384 targets | 0.833538 →0.844838 | 0.018833 →0.017945 |
| Frozen-codec input, same384 | 0.786924 →0.797697 | 0.027989 →0.026743 |
| Clean player HP0,18 targets | 0.602691 →0.714730 | 0.035692 →0.026571 |
| Clean navigation,96 targets | 0.898286 →0.900133 | 0.010588 →0.010529 |

The384-frame panel is episode/seed-held-out, not a novel-image guarantee: it
contains3 exact LR+HR image matches to TRAIN and11 within-VAL duplicates. A
supplemental **370 unique images with all TRAIN pixel matches removed**, using
the same saved predictions without new inference, also improved: clean SSIM
**0.833022 →0.844368**, MAE **0.018833 →0.017947**; codec-input SSIM
**0.786926 →0.797814**, MAE **0.027925 →0.026670**.

The overall0.90 SSIM criterion remains **false**. These are SR measurements,
not autonomous navigation/combat qualification or a performance benchmark.
All74 SR tensors changed, Adam reached2000, exported weights were finite, and
the original SR file plus every frozen-codec tensor/buffer stayed unchanged.
The new artifact was **not automatically deployed**.

The user then requested **10000 local refinement updates**, or30000 cumulative
including the20000-step parent. `continuation_2000_to_10000.json` explicitly
extends the budget while the original training plan stays unchanged. Full
checkpoint, export and reports from2000 were archived first. Model, Adam,
private NumPy RNG, Torch RNG and CUDA RNG restored bit for bit; actual update
**2001 / cumulative22001** was saved and verified. This records the start of
the extension, **not its completion**. The same data, frozen codec, batch/LR,
validation100 and preview100 continue at [local SR progress](http://127.0.0.1:8775).
The active game keeps its original20000-step SR until a separate deployment.

Sources: [paired data audit](training_runs/sr_combat_refine_20261111/data_audit.json),
[selected2000 validation](training_runs/sr_combat_refine_20261111/archives/validation_g002000.json),
[strict370 check](training_runs/sr_combat_refine_20261111/archives/strict_unique_validation_g002000.json),
[2000 tensor/Adam proof](training_runs/sr_combat_refine_20261111/archives/final_verification_g002000.json),
[budget extension](training_runs/sr_combat_refine_20261111/continuation_2000_to_10000.json),
[exact restoration](training_runs/sr_combat_refine_20261111/resume_verification_g2000.json),
[first continued update](training_runs/sr_combat_refine_20261111/first_continuation_update.json),
[archived2000 export](training_runs/sr_combat_refine_20261111/archives/sr_for_game_g002000.pt).

## H35. Short autonomous branches, balanced spawns and negative attacks

The requested **1/4/8/16-transition** combat branches start from the main450-step
rollout's own states and actual h/z memory. They never reinitialize from images.
Source supplies training targets only, without correcting gameplay states;
V2 numerical feedback stays detached and memory keeps BPTT16. The bounds are
4 branches and64 Source probes per combat update.

An additional256 TRAIN/64 VAL genuine32-step episodes cover a nearby enemy
behind a corner that cannot attack:1101/302 eligible negative attacks and254/63
later real attacks. The combined corpus has147202 unique TRAIN and36804 VAL
transitions. Explicit role aliases reuse existing long tapes for short spawn
windows; they are not counted as newly collected data. Auxiliary B16 uses
**4 spawn +4 negative-attack +8 legacy-mixture windows**, balancing spawns
across the eight actual anchor locations.

Every250 updates, validation checks **all128 revivals from64 long VAL tapes**
plus all302 negative and63 positive attacks from the new VAL as separate cohorts.
Old short-panel reports remain available; exhaustive spawn checking replaces
the nine-event spawn gate. Navigation, own450, collision and life guards stay.
Baseline4159: spawn **18/128**, negative-attack FPR **1.0**, positive recall
**1.0**; quality failed.

The previous worker stopped gracefully after **4159**. Its complete checkpoint,
Adam and RNG were archived; `--combat-recovery-plan` continues without rollback
or optimizer reset. Verification4159→4160 confirmed1156 bit-exact frozen tensors,
115 changed trainable tensors and exactly one increment for every Adam clock.
The real update used4 H16 branches,10 eligible negative attacks and5 spawn targets;
loss85.1209 and gradient norm45.2327 were finite. This proves execution, not
quality improvement. All10 new CPU tests passed, as did a separate full450-step
backward probe on a disposable model copy without an optimizer step.

Sources: [hard-negative audit](training_runs/combat_hard_negatives_20261111/data_audit.json),
[combined manifest](training_runs/rssm_combat_recovery_v3_20261111/data/manifest.json),
[full450 probe](training_runs/rssm_combat_recovery_v3_20261111/preflight/full450_backward.json),
[V3 plan](training_runs/rssm25m_combat_20261111/combat_recovery_revision.json),
[baseline4159](training_runs/rssm25m_combat_20261111/validation/step_004159/rssm_combat_validation.json),
[exact restoration](training_runs/rssm25m_combat_20261111/resume_verification_g4159_recovery_v3.json),
[first update](training_runs/rssm25m_combat_20261111/first_combat_recovery_update_verification.json).
Dynamics dashboard: http://127.0.0.1:8765/.

## H36. SR continues until strict clean SSIM/MAE quality, without a step cap

The user superseded the10000-update budget: train SR until **mean clean SSIM
strictly greater than0.91 AND mean clean MAE strictly less than2.5/255**
(0.00980392156862745 in normalized RGB). The2.5 unit follows the dashboard's
MAE display, which multiplies normalized MAE by255. Both metrics use the same
fixed384-frame episode-held-out Source32→HR128 panel from H34, including its
documented duplicate-image limitation. Codec-degraded input and navigation
regression checks remain separate reports; they are not silently substituted
for the requested clean criterion or gameplay qualification.

The SR worker stopped gracefully at **8672 local updates**, and its full Adam
checkpoint was archived. Immutable `quality_revision_ssim091_mae2p5.json` binds
that checkpoint, the original training plan, prior10000-budget ledger and
validation-panel fingerprint. The previous budget/history were preserved.
Weights, Adam, NumPy/Torch/CUDA RNG, data, frozen codec, batch8 and LR`5e-5`
restored unchanged. The first real continued update **8673 / cumulative28673**
was saved. The numerical restoration proof compares model and all Adam/RNG
fields exactly; codec equality is measured at validation, with its precise
measurement step recorded rather than assumed for an unmeasured first tick.

`--quality-revision-plan` overrides the historical CLI step budget. There is
**no fixed update/time cap or finite ETA**; validation and previews remain
every100 updates. Equality, missing metrics, nonfinite values, wrong input
domain or incomplete panel coverage cannot pass. One deterministic check
meeting both strict clean means stops the run and exports those qualifying
weights; a worse or differently qualified historical checkpoint cannot replace
them. Nothing is automatically deployed to the game.

| Pinned check | Clean SSIM | Clean MAE×255 | Requested criterion |
|---|---:|---:|---|
| Quality-revision baseline8672 | 0.851791 | 4.434637 | false |
|10600, beyond the former10000 cap | 0.853023 | 4.447718 | false |

The target had **not** been reached at10600. Training remained active; this is
not a claim that either metric improves monotonically or will reach the target
within a known time. The active game still used its original20000-step SR.
The live [SR dashboard](http://127.0.0.1:8775) shows the clean strict goal and
codec-degraded measurements separately, without a misleading10000 denominator.

Sources: [immutable quality revision](training_runs/sr_combat_refine_20261111/quality_revision_ssim091_mae2p5.json),
[exact8672 restoration](training_runs/sr_combat_refine_20261111/quality_resume_verification_g8672.json),
[first8673 update/provenance](training_runs/sr_combat_refine_20261111/first_quality_revision_update.json),
[baseline8672](training_runs/sr_combat_refine_20261111/validation/step_008672.json),
[check10600](training_runs/sr_combat_refine_20261111/validation/step_010600.json),
[quality launch command](training_runs/sr_combat_refine_20261111/quality_launch.json).

## H37. User-requested SR stop and the latest paired game

Following the new request, SR stopped at **18788 local updates**
(38788 cumulative); its full last checkpoint with Adam was saved. The new game
test uses the separate **best18100** (38100 cumulative), not the latest weights.
On the same clean384-frame panel it achieved SSIM **0.856338** and
MAE **0.017128**, or **4.367513/255**. The H36 conditions—SSIM>0.91 and
MAE<2.5/255 simultaneously—**were not met**. A user-requested stop does not
turn the validation result into a pass. Codec-degraded results remain separate.

The H33 neural renderer naturally completed **6000 updates** of its corrected
phase. On the final256-frame panel, Base achieved SSIM0.753084/MAE0.031992;
RF1 achieved SSIM0.752190/MAE0.033224, with RF1 NPC MAE0.097196. Base passed the
strict no-regression guard against its original parent, but the condition
that RF must be no worse than the current Base across all required cohorts
**did not pass**. The full Source and release gates therefore remain false.
Completing6000 updates does not establish a complete quality fix. The trained
compact VQGAN1900, map, RSSM and SR remained frozen in this worker.

The user requested a game test of the latest results. A separate session was
created at `training_runs/paired_compare_20261111T181714Z_latest_sr`:

| Component | Weights actually attached |
|---|---|
| Dynamics | RSSM snapshot4250; main training continues independently |
| Base / RF1 | Latest6000; cumulative Base15200/RF15600; one Euler step |
| Codec | Frozen compact VQGAN, selected1900; its previous quality false retained |
| SR | Best18100; full last18788 stored separately |
| Resolution | Native32 → learned SR128; then bilinear for display |

Exporting Base/RF preserved every tensor from the full6000 checkpoint.
Strict reload checked Base, codec, normalization, noise and the original
renderer assets. The explicit navigation→combat rebind used two profile views
of **the same4250 world tensors**, rather than different training checkpoints.
Every world tensor was bitexact across the profile change. The bundle retains
`source_quality_passed=false`, `release_passed=false`, `gameplay_qualified=false`;
`--allow-unqualified-codec` permits only the requested experimental test and
does not certify image fidelity.

At **21:24:57 Moscow time on11 November**, Pygame process60616 actually opened.
It uses **FP32**; the prepared command containing untested FP16 was superseded
by the actual launch command. The1572×968 window has two768×768 panels fitted
to the available work area. Its screenshot was inspected. A short GPU preflight
passed navigation step → E/combat → shot → return to navigation and explicit
reset. This verifies mode operation, not450-step autonomous gameplay or30FPS.

Startup is **navigation without an enemy**, in **shared_camera** mode with
`shared_camera_entities=neural`. Only the Source player's camera coordinates
and direction are substituted for display. NPC, HP and memory remain neural;
the model's own state/h/z are not corrected from the Source camera. **Tab**
selects the fully autonomous camera. **E** switches navigation/combat while
preserving player/time/h/z and explicitly initializing the single NPC slot at
the scene boundary. **W/S/up/down** move, **A/D** strafe, **Q/left/right** turn,
**Space/left click** fire, **P** pauses, **R** explicitly resets the trial and
**Esc** exits. Both views receive identical actions and dt. Shared-camera
results are not presented as autonomous trajectory agreement.

Sources: [SR stop and selection](training_runs/sr_combat_refine_20261111/session_export.json),
[SR18100, clean and codec separately](training_runs/sr_combat_refine_20261111/validation/step_018100.json),
[NPR6000 and failed guards](training_runs/renderer_repair_corrected_20261111/validation/step_006000.json),
[verified6000 export](training_runs/paired_compare_20261111T181714Z_latest_sr/renderer_export_verification.json),
[explicit profile rebind](training_runs/paired_compare_20261111T181714Z_latest_sr/renderer_profile_rebind.json),
[actual FP32 launch command](training_runs/paired_compare_20261111T181714Z_latest_sr/launch_command.json),
[GPU preflight](training_runs/paired_compare_20261111T181714Z_latest_sr/preflight.json),
[opened-window screenshot](training_runs/paired_compare_20261111T181714Z_latest_sr/window_ready.png).

## H38. T for combat, E for turning and combat neural-game validation

The requested controls are now separate: **T** switches navigation/combat,
**E/right** turns right and **Q/left** turns left. H37 preserves the historical
E toggle. This changes the game interface, not trained weights or the training
protocol.

At **21:32:10 Moscow time on11 November**, the previous window60616 was closed
cleanly with its report retained, and a new visible window53996 opened:
`training_runs/paired_compare_20261111T183210Z_tcombat`. It uses the same
immutable RSSM4250, Base/RF6000, codec1900 and SRbest18100 from H37, FP32,
shared_camera with neural NPC/HP, and navigation startup. Three CPU control
tests passed. These facts do not establish combat quality.

Then **three450-transition trials** completed:1350 own NN steps and1350
independent Source steps. They use the actual FP32 game path:
RSSM → Base/RF1 → SR128 → bilinear1024. Source `step/render/set_state_vector`
are forbidden inside `NeuralRuntime.step`; the independent Source advances
after a successful NN step. One initial NAV frame is observed before explicit
NPC activation. State and memory continue without correction; all values are
finite. Both autonomous native RGB and shared-camera images are saved; the
latter substitutes only the display camera, not the dynamics update.

| Trial | Source and own NN events | State/geometry errors |
|---|---|---|
| `front_fire` | Shots65/65 matched; Source6 hits/3 kills/3 respawns, NN2/1/1; same-tick hit/kill/respawn matches0 | NPC life differs on195/450 steps; NPC XY mean1.983697 tiles across177 continuing Source-life transitions |
| `close_attack` | All13 player-damage events and death at433 matched exactly; HP differences are rounding only | NPC XY mean0.716267 tiles across450 transitions; Source active433, post-death hold17 transitions |
| `occluded_close` | Initial negative is correct;1 false attack across3 eligible wall-occluded negatives; later Source13 attacks/death436, NN7 attacks/no death, final HP44 | NPC XY mean0.551607 tiles across450 transitions; NPC overlaps walls for297 steps, first9; swept-wall298 versus Source0 |

Paired event counts mean **Source/NN**; matching requires the same tick, while
NN events at different times are counted separately. In `front_fire`, hit TP0,
FP2, FN6: all-frame accuracy0.982222 comes from442 negative frames and **does
not establish learned combat**. In `occluded_close`, player-hit TP0, FP7, FN13;
FPR1/3 refers only to three Source attack opportunities with blocked visibility,
not all450 frames. Post-death Source holds are separately counted:17 transitions
in `close_attack` and14 in `occluded_close`. They do not replace active combat
negatives. A cohort with no positive events has recall null.

The first two trials are fixed manual scenarios; the third uses a held-out
initial wall pose followed by a new continuous Source trajectory. These are
diagnostic counterexamples, not estimates of the full game population. The
stationary player does not cross walls in these trials, but NPC position,
hits, life and respawns diverge. The result is **mixed, gameplay_qualified=false**:
close damage is reproduced, while reliable combat and NPC motion remain unqualified.

A separate visual check used12 saved frames,11 with visible NPCs. Re-rendering
the actual own conditions was bitexact to the captured images. On Source NPC
pixels, its MAE was0.169862 with presence proxy6/11. Using the **full GT state
and the same own features**, RF achieved MAE0.145541 and proxy9/11;
Base achieved MAE0.137134. This substitution was measurement-only; no GT state
entered the rollout. It neither isolates one coordinate from the other state
fields nor establishes a mathematical quality ceiling.

At `close_attack`128, own NPC error0.858 tiles accompanies the enemy's disappearance
from the image, while the GT-state query draws it. Dynamics alone does not
explain all the error: in distant `front_fire`98, the Source NPC covers5 native
pixels; XY error is4.624 tiles, own ROI MAE0.32710, GT-state RF0.21558 versus
Base0.19530. In HP0 frame `close_attack`450, GT-state RF MAE0.13589 is worse
than Base0.10979. Thus **both conditioning and rendering** have errors. Native
and SR SSIM are not presented as a single improvement: their resolutions and
Source targets differ. This small diagnostic set does not pass the Source gate.

Restart evidence: [actual launch command and T/E/Q](training_runs/paired_compare_20261111T183210Z_tcombat/launch_command.json),
[new-window screenshot](training_runs/paired_compare_20261111T183210Z_tcombat/window_ready.png).
Checks: [protocol and weight SHAs](training_runs/paired_combat_validation_20261111/protocol.json),
[all1350 steps](training_runs/paired_combat_validation_20261111/report.json),
[front_fire](training_runs/paired_combat_validation_20261111/front_fire/report.json),
[close_attack](training_runs/paired_combat_validation_20261111/close_attack/report.json),
[occluded_close](training_runs/paired_combat_validation_20261111/occluded_close/report.json).
Images: [visual report](training_runs/paired_combat_validation_20261111/visuals_fp32/report.json),
[combined comparison sheet](training_runs/paired_combat_validation_20261111/visuals_fp32/combat_comparison.png).

## H39. Distance supervision, effective128 and updated preview assets

After H38, the user requested better mid/far targets, batch128 and the updated
NPR/SR in “What the model shows.” The previous RSSM stopped at4398 with a full
checkpoint; v4 continued **without rollback or an Adam reset**. The first full
4399 was saved and verified:1156 frozen tensors were unchanged,115 trainable
tensors changed and remained finite, and all115 Adam clocks advanced exactly1.
Previous weights, Adam and RNG were restored.

One optimizer update now contains **32 ×4 =128** main episodes of450 transitions
with TBPTT16. All four microbatches use the same environment/context/latent mode.
Gradients are zeroed once, averaged after four complete backwards, then clipped
before one Adam step. Phase clocks advance once, auxiliary draws by128. Losses
are averaged, event/work counts are summed, and all four original rows are kept.
Raw per-microbatch gradients were not saved: executable source order and the
CPU gradient/Adam test verify the reduction; Adam clocks independently establish
one update. Batch-dependent statistics still use32, not globally pooled128
statistics. Train loss4399 was75.563060 and gradient norm13.590139; both were
finite, but neither establishes gameplay quality.

**Source target selection and normalization** were corrected without rewriting
the corpus. Each combat microbatch32 contains8 spawn,8 attack-negative,3 near-hit,
3 mid-hit,3 far-hit,3 fired-miss and4 legacy windows. Target CE uses actual Source
shots only, gives hit/miss means mass1:1 and equally weights present near<2,
mid[2,5),far≥5 groups. These are mean weights, not equal actual event counts.
At4399 there were120 shots:51 hit+69 miss;1928 nonshot positions out of2048 were
excluded from target CE. All48 specifically selected target windows contained
genuine future Source events. They are targets/selection labels, not future
prior inputs. The new loss total is not directly comparable to v3.

An immutable file-SHA plan supplies NPR6000 and SRbest18100 to the preview.
A separate frozen clone uses the **current RSSM**, not the game's4250 snapshot;
training Base/Flow/VQGAN/SR and optimizer are not replaced. Initialization4398
verified unchanged training tensors/parameter IDs. RNG streams are protected
during construction and every preview. NPR has cumulative15600, Base15200 and
SR38100 (local best18100); quality/release false remains recorded. Five rows:
context → Source future for comparison → own RGB32 → SR128 → bilinear512.
H4 is not a450-step video. Preview cadence remains10 optimizer updates and
full validation250. The UI separately shows the committed optimizer step,
completed microbatches and current450-transition progress. The H38 game window
and its weights were not replaced.

A separate renderer inventory found715 far examples among the previous16000
draws, with median NPC area3 native pixels. The opt-in sampler assigns25% near,
25% mid,25% far and25% original replay. Positive examples require a living NPC
and nonempty exact Source mask. Previous weights, Adam/RNG,96000 parent draws
and the replay mixture were retained; selection does not provide Source RGB
to inference. An independent static VAL panel contains192 images,64 per band;
the original256 and HP0 remain separate guards.

Two matched6000→6200 pilots used the same200 minibatch orders, Adam/RNG and
frozen codec. They compared distance-only against the same sampler with
crop-SSIM weight0.2 on the native bbox+2 pixels, without resizing the target:

| RF foreground MAE, fixed192 | Parent6000 | Sampler +crop0.2 | Sampler only |
|---|---:|---:|---:|
| Near |0.128461|0.130528|0.127175|
| Mid |0.140459|0.143813|0.141425|
| Far |0.111051|0.112984|0.110338|

Crop0.2 was **rejected** because RF worsened at every distance. Sampler-only
slightly improved near/far but worsened mid; Base near was almost unchanged,
far improved and mid worsened. The strict gate requires foreground improvement
for **both heads at all three distances** greater than0.0005, with no full-MAE
regression beyond0.002. It did not pass, nor did the overall release gate;
active game assets were unchanged. Sampler-only **actually continued6200→8000**
with cropOFF and preserved Adam/RNG:6201 executed, and saved6290 verified every
Adam clock advancing6200→6290, sampler draws99200→100640, and all128 codec
tensors/buffers bitexact. This compares saved states, not a new measurement of
historical pre-update GPU memory; exact moment loading is additionally covered
by a CPU test with a bitexact next Adam step. Monitor8774 follows the new worker;
the game bundle was not switched.

At6400 RF improved at every distance, but Base near/mid still failed the required
improvement. At the next fixed check, **6600**, the relative distance gate
**first passed for both heads at every distance**:

| Foreground MAE, fixed192 | Base6600 | RF6600 |
|---|---:|---:|
| Near |0.128832|0.126859|
| Mid |0.139630|0.137517|
| Far |0.102559|0.104171|

This is relative improvement over parent6000 on the same panel, not Source
quality or gameplay qualification: the full Source and release gates remain
**false**. Training continues toward8000; neither reliably visible NPCs nor a
complete mid/far fix is claimed, and game assets are not replaced automatically.

The later dual-RSSM launch activated a **preview-only6600 revision**. The original
6000 plan and its SHA remain immutable in the history; a separate descriptor
selects6600. Its RF cumulative count is16200, Base15800, and SR remains best18100,
cumulative38100. Initialization at4430 verified unchanged training tensors,
parameter IDs and optimizer; the live4440 preview used current dual dynamics,
five rows and no future Source feedback. Source/release/gameplay remain false.
The user game and its assets were unchanged. See the
[revision](training_runs/rssm_dual_20261111/preview_assets_revision_npr6600.json),
[selected plan](training_runs/rssm_dual_20261111/preview_assets_npr6600_sr18100.json)
and [clone proof](training_runs/rssm_dual_20261111/updated_preview_assets_verification.json).

Sources: [v4 plan](training_runs/rssm25m_combat_20261111/batch128_revision.json),
[first4399](training_runs/rssm25m_combat_20261111/first_batch128_update_verification.json),
[preview-assets plan](training_runs/rssm25m_combat_20261111/preview_assets_batch128_g4398.json),
[separate-clone verification](training_runs/rssm25m_combat_20261111/updated_preview_assets_verification.json),
[distance inventory](training_runs/renderer_distance_refine_20261111/source_distance_inventory.json),
[matched pilots](training_runs/renderer_distance_refine_20261111/pilot_comparison.json),
[6200→8000 continuation](training_runs/renderer_distance_refine_20261111/training_rebalance_only/budget_extension_006200_008000.json),
[actual6290](training_runs/renderer_distance_refine_20261111/training_rebalance_only/continuation_actual_verification.json),
[check6400](training_runs/renderer_distance_refine_20261111/training_rebalance_only/validation/step_006400.json),
[relative gate6600](training_runs/renderer_distance_refine_20261111/training_rebalance_only/validation/step_006600.json),
[CPU accumulation](tests/test_rssm_accumulation.py), [CPU preview](tests/test_rssm_preview_assets.py),
[independent distance/crop audit](tests/test_renderer_distance_audit.py).

## H40. An independent enemy RSSM and verified dual continuation

The user requested a separate compact recurrent model for the enemy while
retaining the original player RSSM. The previous worker stopped at a complete
4430 boundary. A new run, `training_runs/rssm_dual_20261111`, appended only
`vjepa_backend.enemy_rssm.*`; all previous model tensors, Adam groups/moments,
RNG, curriculum, data bindings and batch128 clocks were preserved. The new enemy
group started with empty Adam state. This is an architecture extension, not a
restart of the player or a claim of identical NPC predictions.

| Component | Parameters | Packed FP32 memory |
|---|---:|---:|
| Retained dynamics including shared map/observation decoder |25007048|2032 =1008+32×32|
| New independent enemy RSSM |2438019|640 =384+8×32|
| Combined dynamics |27445067|2672|

Trainable dynamics total25725353; frozen image renderer/SR are excluded from
these dynamics counts. The enemy has its own recurrent/categorical state;
renderer appearance features remain player-derived. Existing game-specific
typed state and known action/clock rules remain. No actor/critic, EMA or online
Source correction was introduced. Main training stays32×4=128,450 own
transitions with TBPTT16; Source posterior supervision is separate from the
independent prior.

Actual update4431 was recorded in metrics, but no checkpoint4431 was saved.
The independent CPU audit therefore uses the **first available persisted4440**,
exactly10 updates after4430. All1156 frozen tensors are bitexact. All50 new
enemy parameter tensors changed and are finite; all50 new Adam states have
clock10. Old115 Adam states advanced by10 for98, by8 for13 and by0 for4;
unused states remained bitexact. Two optimizer groups and the old name/ID
mapping are preserved. Auxiliary draws advanced1280. Initial old tensors,
moments and RNG were checked against the source archive and actual resume
proof; the fresh enemy initialization was independently reconstructed with
seed173. No raw gradients were saved or claimed as measured.

The local8765 page showed the dual caption,27,445,067 parameters, memory2672,
batch32×4 and the6600/18100 preview revision. Reusing old image bundles requires
explicit migration lineage plus strict Base/map/appearance/codec/norm/noise
identity checks. The original6000 preview pin remains recorded. This verifies
real training and compatible frozen assets, not combat or15-second qualification.
The already open game continues using its earlier4250 snapshot.

Sources: [activation](training_runs/rssm_dual_20261111/activation.json),
[migration](training_runs/rssm_dual_20261111/checkpoints/rssm_world.pt.migration.json),
[actual restore](training_runs/rssm_dual_20261111/resume_verification_g4430_batch128_v4.json),
[persisted4440 proof](training_runs/rssm_dual_20261111/first_dual_update_verification.json),
[saved launch](training_runs/rssm_dual_20261111/launch_command.json),
[migration CPU tests](tests/test_dual_rssm_migration.py).

## H41. Ordered tiny-enemy codec, SR and renderer training

The user requested sequential improvement of small enemies throughout the image
pipeline. A new immutable run, `training_runs/small_enemy_pipeline_20261111`,
uses **codec → SR → Base/RF1**, monitored at **http://127.0.0.1:8776/**. The
parents are world4250, compact codec1900, renderer8000 and SRbest18100. This
training does not replace the active game or alter dual dynamics.

The new data contains4608 true Source32/128 pairs, with exact native/HR masks:

| Split | Total | Living tiny NPC,1–16 native pixels | Navigation | Near | Corpse | HP0 |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN |4096|3072|317|317|317|73|
| VAL |512|384|37|36|37|18|

Independent checks verified selected target RGB, parent episode seeds/files and
health-aware render-condition keys do not overlap between splits. Past observed
context can coincide; this is not a claim that every context frame is unique.
All target images within each split are unique; tiny NPCs are genuinely alive
and visible at native32. HR128 is a reference-engine render at the same physical
tick, not interpolation. Source replay and exact NPC-removal masks were audited.

Codec E/D **completed2000 and selected2000**. Tiny Source reconstruction
foreground MAE improved0.08717198→0.05840806. The unchanged-parent full RF1
reference changed0.10076094→0.10153928, within its explicit compatibility
tolerance; this is not an RF improvement. The relative stage gate passed,
including navigation/near/corpse/HP0 guards. Full Source and gameplay quality
remain separate. Q and latent normalization remain fixed. SR then actually
started; renderer remains queued at this entry, not trained by the readiness
test. SR retains its1517571-parameter EDSR-like network; Base996844/RF6992192
remain the later-stage architecture, with codec/SR frozen during renderer work.

Each initial milestone is2000 updates. Failure extends the same stage by1000
with its checkpoint/Adam/RNG; it must not skip ahead. Independent CPU tests
reproduced and then verified a fix for an early passing selected report that
could bypass a still-running codec milestone. Advancement now also requires
durable completed status, the full milestone, no live child, and a matching
selected artifact/report/panel. Controller restarts verify PID and actual
command before adopting a live child, preventing duplicate workers. Missing or
nonfinite cohort values reject the gate. This relative gate requires tiny
foreground improvement greater than0.0005, with MAE tolerance0.003 and SSIM
tolerance0.02 for forgetting guards; it is not an absolute game-quality pass.

Independent readiness tests used real32→128 forward/backward on fresh/private
copies: SR and Base/RF gradients were nonzero, all frozen codec tensors/buffers
were bitexact, and optimizer steps were0. The selected clean/codec SR panel,
native-pixel cohort definitions, true HR target, RNG/modes, absent/nonfinite
metrics and image-preview shapes also passed CPU tests. No game promotion or
new training was performed by these tests.

Sources: [ordered plan](training_runs/small_enemy_pipeline_20261111/ordered_plan.json),
[data audit](training_runs/small_enemy_pipeline_20261111/data_audit.json),
[Source replay](training_runs/small_enemy_pipeline_20261111/data_source_replay_audit.json),
[parent assets](training_runs/small_enemy_pipeline_20261111/parent_assets.json),
[codec selected2000](training_runs/small_enemy_pipeline_20261111/codec/validation_selected.json),
[codec handoff](training_runs/small_enemy_pipeline_20261111/codec/ordered_stage_pass.json),
[CPU readiness](training_runs/small_enemy_pipeline_20261111/future_stage_readiness_cpu.json),
[controller tests](tests/test_small_enemy_pipeline.py),
[SR tests](tests/test_small_enemy_sr.py).

## H42. Separate Player, Enemy, Interaction loss normalization

The user explicitly requested separate normalization, balancing and display of
the three tasks. The dual worker stopped at a complete4454 boundary. The new
run `training_runs/rssm_dual_balanced_20261111` continued from the full snapshot
without an architecture change, rollback or reset of Adam/RNG. The immutable
`--loss-balance-plan` is mandatory for subsequent resumes. The model, prior
data/sampler clocks,32×4=128 update,450 own transitions, TBPTT16, validation250,
preview10 and preview assets NPR6600/SR18100 remain. The open game is untouched.

Player contains player motion/contact/XY/wall objectives and its posterior
KL/reconstruction. Enemy contains NPC motion/XY/walls/spawn and its own KL.
Interaction contains hit/target/attack/damage/death and the remaining combat
state fields. These are task partitions, not three new copies of each loss.
Raw group sums retain the previous internal component coefficients and exclude
constant KL floors. Each task has priority1. Its scale is the previous committed
EMA, decay0.95 with minimum0.1, in separate combat/navigation banks. The scale
is fixed across every TBPTT chunk, short branch and accumulated microbatch in
one optimizer update. EMA changes once after successful Adam commit; failed
or incomplete updates do not advance it.

The page shows literal **Player loss / Enemy loss / Interaction loss** as
weighted contributions, with raw/scale, normalized, effective weight and floor
below. The formula is **weighted =raw/scale ×priority +KL-floor**. The constant
floors1.1 for Player and1.1 for Enemy remain unscaled, outside EMA and with no
gradient; Interaction floor is0. Thus total is the sum of the three displayed
weighted contributions, including floor2.2. Effective weight is priority/scale.

The first update in each environment bank calibrates independently at scales1;
this is not a single global calibration. The first saved combat update4455
calibrated its bank. The next saved4456 actually
used the preceding4455 raw magnitudes:

| Group | Raw4455 | Weighted4455 | Scale4456 | Raw4456 | Weighted4456 |
|---|---:|---:|---:|---:|---:|
| Player |14.20850038|15.30850038|14.20850038|16.64843654|2.27172369|
| Enemy |54.13452294|55.23452294|54.13452294|56.37695629|2.14142335|
| Interaction |7.34662673|7.34662673|7.34662673|6.88350053|0.93696070|

Total77.88965005→5.35010775 is principally a **change in loss units**, not a
measured quality improvement. Effective weights at4456 were respectively
0.07038040/0.01847250/0.13611689. No new held-out navigation/combat pass is
claimed. The previous total curve is excluded under the new protocol
`rssm_dual_group_balance_v5`; missing or stale step/protocol values show a dash.

The4454→4456 CPU audit verified all1156 frozen tensors bitexact, both RSSM cores
updated,161 active Adam states advanced twice and4 dormant legacy attack-head
states/weights remained unchanged. All165 states, two groups, finite moments,
EMA and sampler clocks were checked:2 combat bank updates and0 navigation bank
updates at that snapshot. Initial restore compared1818 tensors and
model/Adam/NumPy/Torch/CUDA RNG exactly. The local8765 HTML matched the current
source bytes, and the actual served script rendered current groups and their
sum correctly. No dashboard restart was necessary. The three labels, real zero,
missing/nonfinite values and stale protocols passed the60 Node tests and build.
These checks establish the implementation and preserved state, not successful
15-second gameplay or equal gradient norms for all three tasks.

Sources: [revision plan](training_runs/rssm_dual_balanced_20261111/loss_balance_revision.json),
[initial restore](training_runs/rssm_dual_balanced_20261111/resume_verification_g4454_groupbalance_v5.json),
[saved4456 audit](training_runs/rssm_dual_balanced_20261111/first_group_balance_verification.json),
[live UI proof](training_runs/rssm_dual_balanced_20261111/dashboard_group_loss_verification.json),
[saved launch](training_runs/rssm_dual_balanced_20261111/launch_command.json),
[loss-balance tests](tests/test_rssm_loss_balance.py),
[dashboard tests](training_dashboard/worker.test.mjs).

## H43. A20M enemy RSSM and temporary combat-only training

The user temporarily removed no-enemy navigation from training, then requested
an enemy RSSM of about20M parameters. The previous balanced dual worker stopped
at a complete4482 boundary. New run `training_runs/rssm_enemy20m_combat_20261112`
retained the player/shared weights, their Adam, RNG and history; the enemy was
resized through an explicit migration rather than an unrecorded model reset.
The old checkpoint remains archived. The first full four-microbatch combat
update4483 completed and was saved; training continued afterward.
At4485, a slot that the old schedule would have assigned to navigation, all
four microbatches actually used combat, context1 and argmax. Historical
navigation updates152 and auxiliary draws4784 did not advance. Thus4482→4485
includes a measured check that the old navigation slot was removed, not merely
a new flag or renamed phase.
The independent checkpoint audit used saved4486 and checked all four updates
4483–4486 as combat32×4/450, with navigation counters and its EMA bank unchanged.

| Component | Parameters | Packed FP32 memory |
|---|---:|---:|
| Retained player dynamics including map/observation decoder |25007048|2032 =1008+32×32|
| Enlarged enemy RSSM |20000323|1600 =1344+8×32|
| Combined dynamics |45007371|3632|

Enemy hidden width is1408, deterministic width1344; stochastic8×32, embed128,
action128,4 blocks and unimix0.01 remain. Trainable dynamics total43287657;
frozen image renderer/SR are excluded from these dynamics counts. The resize
preserved1271 player/shared tensors exactly. Of50 enemy tensors,21 compatible
tensors were copied exactly;29 resized tensors used documented semantic/channel overlap with
new channels initialized at seed173. The old and enlarged enemy outputs are
not claimed to be identical. Optimizer IDs/groups are preserved:136 existing
states, including exact-shape enemy states, remain;29 resized enemy states
start fresh on their next update. Thus this does **not** preserve every old
enemy Adam moment, and does not reset the retained player.
After four completed updates,132 retained active states advanced by4,4 dormant
states remained unchanged, and all29 new states had clock4.

The immutable `--combat-only-plan` changes the effective update cycle to
`[combat]`, active environments to `[combat]` and navigation replay to false.
Both Player and Enemy still learn within combat episodes; this is not a
player freeze. Historical navigation replay152 and its auxiliary/EMA history
are retained, but no new no-enemy training or navigation-bank EMA commits are
scheduled. Navigation validation remains a forgetting check. Batch32×4=128,
450 own transitions, TBPTT16, Source branches/targets, validation250 and
preview10 remain. Prior state/memory receive no future Source corrections.
The three-task normalization protocol `rssm_dual_group_balance_v5`, its banks
and the unscaled KL floors remain; this schedule revision does not invent a
new loss unit. First4483 loss5.53617249 and gradient norm5.14153475 were finite,
but are not a new held-out navigation/combat pass.

The8765 page actually showed20000323 enemy parameters,45007371 dynamics
parameters and memory2032+1600=3632 from live metadata. The effective schedule
explicitly says combat-only, marks old navigation counters as historical and
keeps navigation validation separate. Missing or contradictory schedule fields
do not claim an active revision. All63 Node tests and build passed.

Old compact Base/RF bundles still require strict image-asset hashes. Reuse now
validates the original append migration followed by the explicit resize;
only enemy deterministic/hidden widths may differ. Player/shared/frozen-asset
proof flags, complete source/target config fingerprints, exact conditioning
identity and unchanged codec/norm/noise are required. Loader and both preview
clone paths retain the resize report. Three new real Base/RF compatibility tests
and the existing focused regressions passed56 CPU tests. The frozen preview
still uses NPR6600/SRbest18100; no game assets or window were replaced. Increased
capacity, a finite first step and preserved assets do not qualify15-second play.

Sources: [combat-only plan](training_runs/rssm_enemy20m_combat_20261112/combat_only_revision.json),
[resize report](training_runs/rssm_enemy20m_combat_20261112/checkpoints/enemy20m_initial_g4482.resize.json),
[initialization](training_runs/rssm_enemy20m_combat_20261112/initialization.json),
[actual restore](training_runs/rssm_enemy20m_combat_20261112/resume_verification_g4482_combatonly_v1.json),
[migration and completed-update audit](training_runs/rssm_enemy20m_combat_20261112/enemy20m_combat_verification.json),
[live dashboard proof](training_runs/rssm_enemy20m_combat_20261112/dashboard_enemy20m_combat_verification.json),
[saved launch](training_runs/rssm_enemy20m_combat_20261112/launch_command.json),
[focused CPU results](training_runs/diagnostic_tools/enemy_resize_renderer_focused_tests.json),
[resize renderer tests](tests/test_enemy_resize_renderer.py).

## H44. Paired game with the20M enemy and updated tiny-image assets

The paired session `training_runs/paired_compare_20261111T221323Z_enemy20m_tiny`
actually opened, ready, as PID51604: Source left, neural output right, initially
**combat / autonomous / FP32**. Snapshots: world4520 with enemy20000323 parameters
and memory3632, renderer selected8300/cumulative17900, compact codec11800, SR
selected2000/cumulative40100. Display: native32 → learned SR128 → bilinear1024
→ fitted panel. Strict NAV→combat rebinding preserved image tensors and
codec/normalization/noise; background training was untouched.

The shape-only preflight used one NAV-initialized step, separately from the
actual combat GUI trial. The durable report records450 paired steps/15.0
simulated seconds, independent clocks/states, finite state/memory, zero resets
or engine corrections and no future Source feedback. This does not qualify
navigation or combat. The final status is **closed**, trial complete; no reason
for closing is inferred, and the window was not relaunched afterward.
[Actual launch](training_runs/paired_compare_20261111T221323Z_enemy20m_tiny/launch.json),
[trial report](training_runs/paired_compare_20261111T221323Z_enemy20m_tiny/report.json).

## H45. Read-only loss audit and fresh validation4930

A read-only audit compared the first64 and last64 updates with matched
context/latent-mode composition. Raw loss77.6267→66.6941 decreased14.08%:
Player XY13.5768→13.07765, down3.68%, and NPC XY54.9024→45.8159, down16.55%.
Normalized total4.9845→5.2088 reflects the moving EMA denominator, not a fixed
convergence scale. These are training objectives, not measured gameplay accuracy;
the plot and its metadata keep the conditions and smoothing explicit.

Fresh validation4930 used the same panel fingerprint as4750. On the450-step
panel, Player XY5.418→4.2604 improved, NPC XY6.268→6.2543 barely changed, and
wall steps3799→4018 worsened. Spawn correctness43/128→39/128 declined. Melee
true positives17/63→47/63 increased, but false positives7/302→60/302 also rose.
The result is mixed and does not qualify gameplay. Training settings, weights,
optimizer and running processes were not changed by this analysis.

Code review found a concrete supervision gap: own-memory branches unroll16
steps, but direct event-head supervision uses step1 only. In updates4867–4930,
their205 respawns supplied only1 directly supervised spawn, and69 attacks
supplied0 direct attack positives. Later state/motion losses and the separate
global Source auxiliary16 remain active; this is not a claim that those heads
receive no training at all. Spawn decoding uses argmax/detach, blocking its XY
gradient to the spawn head. These are supported candidate limitations, not a
proven sole cause or an implemented fix. [Statistics](training_runs/dynamics_loss_audit_20261112/loss_statistics.json),
[fresh validation](training_runs/dynamics_loss_audit_20261112/validation_step_4930.json),
[findings](training_runs/dynamics_loss_audit_20261112/analysis_findings.json).

## H46. V6 combat supervision and independent component normalization

Run `training_runs/rssm_components_v6_20261112` actually continued from4944,
preserving model, all165 Adam states, RNG and actor EMA. Four changes are pinned
by mandatory `--combat-v6-plan`: direct conditional Source head targets cover
all legal own-branch steps1–16, rather than only1; spawn/recovery have independent
normalization; own branches increase from at most4 to8 with first-contact/
post-spawn priority and aligned local player contact; each auxiliary32 selects
8 spawn,8 negatives (6 occluded+2 other),8 positives (4 wall+4 open),8 shooting.
Main450/TBPTT16 and combat-only32×4 remain. Immutable corpora are retained.
Source branch outputs are labels, not corrections to the main prior. Spawn
argmax/detach remains; no differentiable spawn decoder or inference architecture
change was implemented.

Actor BASE uses the inherited actor EMA. Each component uses its own EMA and
priority; its weighted contribution bypasses BASE normalization. Therefore
`actor weighted = actor priority × base_raw / actor scale + sum(component weighted) + KL-floor`,
with `component weighted = priority × raw / component scale`. Actor scale and
effective weight refer only to BASE. A one-microbatch gradient probe with zero
optimizer steps selected spawn-Enemy0.0493519002 and recovery Player0.0311118002/
Enemy0.00309895852/Interaction0.00800152902. Targets were20% for spawn and15%
for each recovery relative to the measured base gradient in that probe, not
percentages of total loss. Minimum priority0.001 replaced a rejected0.05 floor;
the rejected report remains saved. Old actor EMA/Adam/RNG were retained, only
new component banks seeded. KL floors1.1+1.1 remain unscaled. UI v6 formulas and
component rows passed66 Node tests; the old v5 display remains supported.

Saved4945 is verified: four microbatches,32 own branches,512 direct attempts/
334 valid labels, including3 late respawn CE targets and2 actual attack positives.
The128 auxiliary windows contain32 each spawn/negative/positive/shooting.
Loss4.854975345 and gradient norm2.60344 were finite; all1156 frozen tensors
remain bitexact. Baseline4944 was evaluated **before SGD**: close-wall recall51/63
and false positives60/302 are baseline, not v6 improvement. The UI shows both
rates and validation step. New control checks after100/250 updates and global250
checks are scheduled; they have not yet established improvement. Total has
new component-normalized units and is not directly comparable with v5. In
parallel, the requested paired game opened from immutable world4944, reached
450 steps and closed; it was not reopened or altered by this training.
[Plan](training_runs/rssm_components_v6_20261112/combat_v6_revision.json),
[calibration](training_runs/rssm_components_v6_20261112/calibration.json),
[saved first-update proof](training_runs/rssm_components_v6_20261112/first_v6_update_verification.json),
[live UI proof](training_runs/rssm_components_v6_20261112/dashboard_v6_component_verification.json),
[paired trial](training_runs/paired_compare_20261112T064826Z_latest_enemy20m/report.json).

## H47. Effective256 and an exact resume after a status-file lock

The requested **64×4=256** actually continued from4981 in
`training_runs/rssm_b256_20261112`. Saved4982 passed the independent audit:
model, all165 Adam states/groups, RNG, actor/component EMA and calibrated
priorities were retained, and1156 frozen tensors remained bitexact. Four
microbatches64 produce one Adam update; batch-dependent terms use64, not pooled256.
The old128 revision remains historical. Protocol v6 and its loss units are unchanged;
resume requires the saved `--batch256-plan` plus earlier revision pins.

Each auxiliary64 has16 each spawn/negative/positive/shooting. The actual first
update used64 recovery branches:512 selection probes are distinct from1024
direct Source-query attempts, of which566 were valid and458 skipped;10 late
respawns received direct labels. These counts prove exercised supervision,
not better combat. An attempted4983 then stopped on a transient Windows status-file
replacement error. Bounded25-second atomic-replace retry was added, and full4982
was restored exactly; HTTP showed healthy running4982 with4983 underway and no
error banner. UI displayed64×4=256 and historical32×4=128 separately;68 Node tests
and build passed. No game window was altered and no new quality pass is claimed.
[First update](training_runs/rssm_b256_20261112/first_batch256_update_verification.json),
[exact resume](training_runs/rssm_b256_20261112/resume_verification_g4982_batch256_v7.json),
[live UI](training_runs/rssm_b256_20261112/dashboard_batch256_verification.json).

## H48. Revalidating dynamics5010 and diagnosing coordinate drift

An immutable5010 copy used the unchanged4944/5000 held-out panels:
16 autonomous450-transition windows without Source state or memory correction.
Navigation, combat,128 spawn events and365 close-melee transitions completed
in92 seconds. All states/memories were finite; the overall gate failed.
Final mean player drift was3.98493tiles, NPC7.01381 (10 common-alive pairs),
with wall overlap in all16 combat windows. Spawn accuracy50/128; attack recall
58/63; false attacks90/302. One-frame no-enemy navigation ended at0.48773tiles.
All7 panel fingerprints matched. At the first step, NPC speed was0.411 versus
0.700tiles/s with34.74° direction error. An early player case slowed before
contact and suppressed the legal sliding axis.
[Full report](training_runs/dynamics_validation_20261112T094936Z/rssm_combat_validation.json),
[comparison](training_runs/dynamics_validation_20261112T094936Z/compactsummary.json),
[trajectory diagnosis](training_runs/dynamics_validation_20261112T094936Z/trajectory_diagnosis.json).

## H49. Coordinate-drift training and memory agreement from the first update

At the user's request, full5014 was retained and copied to
`training_runs/rssm_drift_curriculum_20261112`, keeping64×4=256.
From the first update, a separate autonomous branch aligns Player/Enemy RSSM
categorical priors and normalized h at1/4/8/16 to stopped posterior targets.
Future RGB is exclusive to targets; no observation or Source correction enters
the autonomous branch. This adapts multi-step agreement, rather than reproducing
the complete PlaNet algorithm.

Source NPC speed/direction losses and player per-axis displacement supervision
balance free, blocked and partial cases. XY gradients accumulate within TBPTT16
without restoring the unstable numerical-conditioning Jacobian. Predicted spawn
teleports sever the old NPC chain; direct local targets keep one-step derivatives.
Numerical forward states stay unchanged. The first100 updates use16 transitions,
then32/64/128/256; update500 uses450. Validation always uses450, including a
pre-SGD baseline and local100/200/300/400/500 controls. Best full model/Adam
weights are selected by450-step drift using Source life masks, so a model-missing
NPC cannot hide its coordinate error. Overall qualification remains separate.

CPU checks and real discarded B64 GPU probes at16/450 passed: finite gradient
norms3.41718/3.45802, peak allocated1.826/2.648GiB. Probes committed no optimizer
or EMA updates; model and all165 Adam states/options remained unchanged.
Exact5014 restoration passed. Worker54164 completed the full5014 baseline and
saved first update5015: four64-example microbatches, H16,4096 main transitions,
four memory backwards and all three kinematic terms active. All1156 frozen
tensors remained exact; both RSSMs changed;161 active Adam clocks advanced once
while four legacy unused clocks remained unchanged. Actor/component EMA committed
exactly once. Loss2.511805, preclip norm3.452078; the shorter horizon changes the
meaning of training loss and does not establish reduced long-horizon drift.
Quality improvement still requires control rollouts.
[Plan](training_runs/rssm_drift_curriculum_20261112/coordinate_curriculum_plan.json),
[GPU probe](training_runs/rssm_drift_curriculum_20261112/coordinates_probe.json),
[exact resume](training_runs/rssm_drift_curriculum_20261112/resume_verification_g5014_coordinate_curriculum_v1.json),
[first saved update](training_runs/rssm_drift_curriculum_20261112/first_coordinate_update_verification.json).

## H50. Rising normalized loss and worse long trajectories

Horizon stayed16 over5015–5052. First/last8-update means: normalized loss
2.55960→2.92260, raw5.26139→4.94665. Reweighting identical recorded terms
with fixed5015 coefficients gives2.52093→2.45834. The plotted rise is therefore
explained by EMA adaptation after450→16; it does not prove each model error worsened.
However, identical full5014/5044 validation shows a real Source-masked drift-score
increase17.2866→21.9058. NPC step16 improved0.1913→0.1481tiles, while player
sliding worsened beyond16 and late own states disappeared from the main training
rollout. The individual causal effects of the new objectives and EMA were not isolated.

Checkpoint5060 was retained before the fix, preserving curriculum clock46/500.
An independent identical16×450 check gave score19.3318, player/NPC mean errors
1.49020/5.94040tiles: recovered versus5044 but worse than historical5014.
This is trajectory-only validation, not a new full quality pass. Best5014 remains retained.
[EMA audit](training_runs/rssm_drift_curriculum_20261112/diagnostics/loss_ema_g5052/loss_ema_diagnosis.json),
[5014/5044 trajectories](training_runs/rssm_drift_curriculum_20261112/diagnostics/drift_worsening_5014_5044.json),
[5060 baseline](training_runs/coordinate_replay_audit_20261112/baseline_comparison.json).

## H51. Long replay and late own-prefix states

At the user's request to fix drift regression,
`training_runs/rssm_coordinate_replay_20261112` continues full5060.
Every fifth new update uses450 transitions; the others follow the existing
16/32/64/128/256/450 schedule. Clock46/500, model,165 Adam states, RNG and old
counters are retained. Histograms and transition counts use actual horizons.
Short/450 EMA values are separated: short starts with exact5060 statistics,
long with the checked5014 EMA. Only normalization statistics are transferred;
model weights do not roll back. Each complete64×4 update commits Adam/EMA once,
leaving the inactive horizon bank unchanged.

On the first short microbatch, eight actual own states and h/z continue to450
under no_grad, without observations, future Source poses or engine correction.
The extra rollout's RNG is restored. Selection reserves8 legal late16-step
branches spanning pre-contact, post-spawn and motion-error states. H450 updates
already supply long own prefixes and need no extra rollout. Missing legal slots
are reported rather than filled through coordinate repair.

The real GPU probe found a B64→B8 map-cache mismatch. Scoped map-cache views and
finally restoration were added, with real endless-spawn64→8→64 and exception
tests. Repeated real B64 probes16/450 passed: gradient norms7.21865/2.38025,
late branches12/16 and16/16, exercising contact and late spawn labels.
Source, model, all Adam and both horizon EMA banks stayed unchanged; optimizer
commits0. Worker59592 restored5060 bitexactly. The first full batch then exposed
a NumPy-int in new telemetry: the metrics write after SGD failed, leaving the
durable checkpoint at5060. Counters now use Python int; plain JSON is checked
before optimizer commit, with a NumPy-bool regression test. Worker70360 reloaded
5060, verifying exact model,165 Adam and NumPy/CPU/CUDA RNG; the unsaved attempt
is repeated. Historical best5014 was inherited
separately; no game was automatically changed. Quality uses the same full450
panel and is assessed separately from implementation correctness.
[Plan](training_runs/rssm_coordinate_replay_20261112/replay_curriculum_plan.json),
[probe](training_runs/rssm_coordinate_replay_20261112/replay_probe.json),
[resume](training_runs/rssm_coordinate_replay_20261112/resume_verification_g5060_coordinate_long_replay_v1.json),
[telemetry-failure recovery](training_runs/rssm_coordinate_replay_20261112/metadata_failure_recovery_g5060.json).

The first five saved updates5061–5065 passed independent verification:4×H16,
1×H450, EMA commits4/1,1156 frozen tensors exact,165 Adam retained, clock46→51.
The actual cumulative main count320000 includes the preceding46 updates;
late branches total49 from short scouts and61 from the full rollout.
Validation5065 scored19.8004 (+2.42% versus5060), so improvement was not claimed.
After ten updates5070 scored18.2629 (−5.53% versus5060): mean player drift
1.49020→1.41254, NPC5.94040→5.60012tiles. However final NPC error worsened
6.61512→7.88826, score p9553.8209→57.0478, and missing Source-alive NPC
steps506→668. Player improvement mainly comes from451 fully blocked Source
transitions; free movement and2066 Source sliding transitions remain worse.
Source-slide coordinate error1.23480→1.31936tiles, velocity MAE
0.35218→0.46786tiles/s. This is a partial mean-trajectory gain, not eliminated drift or
a quality pass. Historical5014 remains best by score. Training continues.
[First-cycle proof](training_runs/rssm_coordinate_replay_20261112/first_replay_cycle_verification.json),
[5070 validation](training_runs/coordinate_replay_audit_20261112/post_cycle_g5070/drift_trajectory_validation.json),
[comparison](training_runs/coordinate_replay_audit_20261112/post_cycle_g5070/comparison.json),
[contact cohorts](training_runs/coordinate_replay_audit_20261112/post_cycle_g5070/source_contact_cohorts.json).

## H52. Temporary XY drift-only training

The user requested coordinate drift alone. Full5079 was retained;
`training_runs/rssm_drift_only_20261112` continues it without resetting Adam/RNG,
clock65/500 or replay19. First new5080 correctly uses450, followed by16; future
32/64/128/256/450 horizons are supported. Batch stays64×4=256.
Only smooth-L1 Player/NPC XY in map tiles is supervised, fixed1:1 actor weights,
plus late own-state XY at1/4/8/16 with the existing0.25 branch multiplier.
Source life masks retain errors for model-missing NPCs, excluding exact Source
teleport ticks.

There are no KL, reconstruction, memory-overshooting, state-field, velocity/heading,
wall/pass-BCE, combat/spawn/event objectives or auxiliary RGB forwards. Dedicated
heads and the decoder are frozen. The allowed40,065,236 parameters/122 tensors
receive XY gradients only, including initial observed-context encoding.
All43 additional frozen tensors and their Adam steps/moments remain exact:
grad=None also prevents momentum and weight decay. All165 canonical optimizer
names/groups/states remain checkpointed, including requires_grad=False entries.
Shared-core features can change inputs to frozen heads; frozen weights do not
promise identical combat predictions. EMA does not scale XY and all its values
stay frozen. Old counters advance as optimizer-update ledgers. The old auxiliary
sampler reserves indices only for RNG/ledger compatibility, without loading a
batch or training events; actual auxiliary fragments/queries0. Validation still
measures full450 dynamics without training other objectives.

Baseline5079 uses the same16×450: score20.97706, mean Player/NPC coordinate errors
1.62325/6.43040tiles. CPU tests and real B64 probes16/450 passed: other-objective
calls0, off-head gradientsNone, discarded-copy model/165Adam/EMA unchanged.
Worker56368 restored5079 exactly. Independent saved5083 audit verifies four pure
XY commits:5080H450,5081–5083H16. All122 allowed tensors actually changed with
Adam clock+4;1199 frozen tensors remain exact, including43 new frozen heads and
every corresponding Adam field. Short/long/navigation EMA values are exact,
clock65→69; Interaction/floor0 and auxiliary/posterior/decoder/memory calls0.
First H450 loss3.31025; short losses0.03045/0.01488/0.03396 use a different horizon
and do not establish improved450-step drift. Best5014 is retained; training continues.
[Plan](training_runs/rssm_drift_only_20261112/drift_only_plan.json),
[GPU probe](training_runs/rssm_drift_only_20261112/drift_only_probe.json),
[5079→5083 audit](training_runs/rssm_drift_only_20261112/first_drift_only_update_verification.json),
[5079 baseline](training_runs/drift_only_audit_20261112/baseline_g5079/drift_trajectory_validation.json).

## H53. Loss-gated horizons with ten consecutive updates

The user replaced the fixed horizon budget and forced one-in-five long updates
with a coordinate-loss gate. The active
`training_runs/rssm_loss_gated_drift_20261112` retains full5112 from the stopped
`rssm_drift_only_20261112`: all model weights,165 Adam names/groups/states, RNG,
historical coordinate/replay counters and frozen EMA values. There is no reset.
The former500-update horizon budget and forced H450 cadence are history-only.
The earlier single-pass preparation was archived unused before GPU probing or
training; the executed plan requires **10 consecutive successes**.

The exact main-horizon path is
`16 → 32 → 64 → 96 → 128 → 160 → 192 → 224 → 256 → 288 → 320 → 352 → 384 → 416 → 448 → 450`.
Each success is one **completed64×4 optimizer update**, using its mean train loss
including main and late XY terms at the current H. Strict loss **<0.25** increments
the streak; **≥0.25** resets it. Only the tenth consecutive success advances one
stage and resets its counter to zero. H450 continues as the final stage.
The numerical objective remains pure XY with fixed weights: no EMA scaling,
KL/floor, reconstruction, combat/spawn, wall, velocity or other objective is enabled.
The122 allowed tensors/40,065,236 parameters train;43 dedicated tensors and their
Adam moments/steps remain frozen, while all165 optimizer histories stay saved.
The existing no-grad450 scout only selects short XY recovery branches; it does
not force an H450 optimizer update or increment the loss gate.

“Что показывает модель” generates the current H and displays four offsets spanning
1..H. It refreshes on a stage advance and hides an old-H preview until the new one
is ready. Full450 held-out validation remains separate, before SGD and every250
global optimizer updates. Mandatory `--loss-gated-curriculum-plan` is retained
alongside every legacy plan pin when resuming; the active stop marker is
`rssm_loss_gated_drift_20261112/stop_training`.

The revised consecutive-gate tests, all16 horizon gradient/denominator tests and
preview tests passed. Discarded B64 GPU probes H16/H96/H450 passed with finite
gradient norms0.55141/0.92469/3.17332 and losses0.003823/0.060369/3.074499.
Model,165 Adam states, EMA and Source bytes stayed exact; disabled gradients were
None, optimizer commits0 and the private gate stayed0/10. H450 was a capacity
probe only. The first actual series5113–5122 used H16 throughout: the first nine
updates held H16; update5122 loss0.005570620024911932 advanced toH32 with streak0.
Its new preview generated32 transitions. This confirms the requested transition,
not full450 gameplay quality or elimination of drift.
[Gate plan](training_runs/rssm_loss_gated_drift_20261112/loss_gated_curriculum_plan.json),
[source preservation](training_runs/rssm_loss_gated_drift_20261112/initialization.json),
[GPU probe](training_runs/rssm_loss_gated_drift_20261112/loss_gated_probe.json),
[first consecutive-series audit](training_runs/rssm_loss_gated_drift_20261112/first_loss_gated_streak_verification.json).

## H54. Student renderer training for small enemies and walls

Started `training_runs/renderer_objects_walls_20261112` alongside the unchanged
loss-gated RSSM training. The student Base/RF1 starts from selected renderer 6600,
because its later 8000 continuation was worse. This phase has fresh renderer Adam
and a 2000-update budget, batch 16, validation every 100 and preview every 10.
The feature source is an immutable RSSM 5152 snapshot; compact VQGAN 997539 (E7)
and SR 38100 are also frozen. The live RSSM continues its separate pure-XY schedule.
Training: **http://127.0.0.1:8777/**. Same-pose Source/codec/Base/RF comparisons:
**http://127.0.0.1:8780/**.

The existing parent corpus already provides 146386 TRAIN and 31277 independent
VAL rows. Living visible NPCs include 16988/2840 examples of area 1–16 pixels and
21719/2320 of area 17–64 pixels in TRAIN/VAL. No new collection was needed.
The sampler assigns 25% each to tiny NPCs, small NPCs, far NPCs and original replay.
Foreground/background errors are normalized independently. Native NPC crops add
L 1 weight 0.25 and edge weight 0.1; wall-boundary and wall-edge terms use 0.2 and 0.1.
Exact Source DDA geometry supplies wall labels, excluding NPC and weapon pixels.
Source targets and masks are loss labels; they are not future-image conditions
or inference inputs. The original main VAL 256 and distance VAL 192 remain, with
a supplementary disjoint VAL 256:64 tiny,64 small,64 far and 64 empty frames.

The focused suite passed 75 CPU tests, and an actual batch 16 GPU backward was finite.
The saved local 90 checkpoint then passed the actual-SGD audit: both Base and RF
changed; all 317 Adam states belong to those components;128 codec tensors and
nonparameter Flow tensors remain bit-exact E7. RSSM and SR pins are unchanged.
The first full held-out validation at 100 updates shows limited improvement:

| Check | Before | Update 100 |
| --- | ---: | ---: |
| Main VAL 256 RF MAE |0.0355328|0.0341181|
| Main VAL 256 RF SSIM |0.744158|0.746047|
| Fixed 8 Source-pose RF NPC error |0.150260|0.143220|
| Fixed 8 Source-pose RF wall-boundary error |0.0999496|0.0979165|
| Fixed 8 Source-pose Base wall-boundary error |0.105151|0.0970501|

These selected eight queries improve RF NPC error by 4.69% and RF wall-boundary
error by 2.03%; they do not establish a global quality pass. The codec is unchanged:
its corresponding baseline NPC/wall errors are 0.092925/0.0381032, empirical
reconstruction references rather than mathematical quality ceilings. Native-NPC
and HP 0/Base guards still fail; release and experimental selection remain false.
Training continues toward 2000 updates. No new game assets are deployed and no
autonomous gameplay-quality claim follows from these teacher-forced image checks.

Evidence: [launch](training_runs/renderer_objects_walls_20261112/launch_command.json),
[data inventory](training_runs/renderer_objects_walls_20261112/inputs/parent_native_area_inventory.json),
[GPU backward](training_runs/renderer_objects_walls_20261112/gpu_backward_probe.json),
[actual SGD/freeze audit](training_runs/renderer_objects_walls_20261112/first_native_area_renderer_update_verification.json),
[initial validation](training_runs/renderer_objects_walls_20261112/validation_initial.json),
[validation 100](training_runs/renderer_objects_walls_20261112/validation/step_000100.json).

## H55. Far-wall strata and preserved Adam continuation

The v 5 control completed update 2000. A separate pixel-residual pilot completed
500 updates, but was not selected: the control was better on the actual full
native Source checks and the same eight NPC/wall queries. Before that pilot,
two CUDA migration attempts failed before SGD and did not modify weights.
Migration was corrected to compare matching model modes and gradient flags;
the actual CUDA output-preservation check then passed. The pixel architecture
remains an evaluated experiment rather than the source of the active continuation.

The selected full control checkpoint at 2000 starts
`training_runs/renderer_far_geometry_20261112` with a target renderer counter
of 6000. This is full continuation: Base/RF weights, all 317 Adam states and
moments/options, MAIN PCG at 32000 draws and all four RNG states were verified
exact after restoration. The next optimizer update is 2001. RSSM 5152, compact
codec E7 and SR 38100 remain frozen; live RSSM training continues independently.
The healthy v 6 worker is monitored at **http://127.0.0.1:8777/**; the diagnostic
page remains **http://127.0.0.1:8780/**. Validation runs every 100 and preview every 10.

Far wall faces can occupy only one or two native 32 pixels and disappear inside
the average error of large nearby walls. New loss-only Source DDA labels split
walls into 5–8, 8–12 and ≥12-tile depth bands, with separate present-image/region
normalization. Each present band combines boundary L1 with weight 0.2 and edge,
corner, thin-face and thin-height errors with weight 0.1 each. Thin-face means
a same-plane/material wall segment occupying 1–2 camera columns at depth≥5;
thin-height means a projected height≤2 native pixels. NPC/weapon occlusion is
excluded. These derived DDA bands, masks and target RGB are supervision labels;
they add no oracle to the rendering conditions or runtime. The existing map/state
conditions remain unchanged, and codec reconstruction is an empirical reference.

MAIN batch 16 retains all v 5 NPC/replay quotas and existing objectives. A separate
four-frame auxiliary batch contributes only far-wall losses with weight 0.25,
so it does not turn the original mixture into a uniform batch 20 loss.
Its independent fresh sampler selects from a fixed, bounded 2048-row TRAIN-only
candidate set:271 depth≥12 frames,481 remaining thin-face frames and 255 remaining
depth≥8 frames. These disjoint strata receive equal quotas. Pool construction
and auxiliary reads leave the complete MAIN sampler unchanged; the auxiliary
RNG/counts restore independently. No VAL mining, generated-image mining, full
corpus DDA scan or new Source collection was needed. Existing held-out panels
and all NPC/empty/HP0/Base preservation checks remain.

The combined suite passed 97 CPU tests. An actual CUDA MAIN 16+aux 4 backward
passed with finite Base/RF gradients and combined loss 0.613047, while all model
weights/buffers remained unchanged with zero optimizer steps; peak allocated
memory was 0.614 GiB. This proves the training path runs, not that distant wall
polygons are repaired.

The first scheduled validation at 2100 uses the same main 256 fingerprint as the
2000 baseline. It shows a tradeoff rather than a global pass:

| Main held-out panel, RF1 | Baseline 2000 | Update 2100 |
| --- | ---: | ---: |
| Full-image MAE |0.0331371|0.0338718|
| Full-image SSIM |0.755794|0.745421|
| NPC foreground MAE |0.0948809|0.0967419|
| Depth≥12 boundary L1 |0.169387|0.130909|
| Depth≥12 thin-face L1 (5 frames) |0.101501|0.0733054|
| Depth≥12 thin-height L1 (14 frames) |0.0766423|0.0648545|

The distant-wall terms improve on this panel, but full-image/NPC quality worsens.
Other native-area guards still report thin-wall regressions, and NPC/HP0/Base
checks fail. Experimental selection and Source/gameplay release remain false;
these results do not justify replacing game assets or claiming general repair.
The deep-wall support is only 35 exposed native pixels on this panel.
The balanced comparison is also available in the diagnostic page and
[its JSON report](http://127.0.0.1:8780/far_geometry_first100.json).

An independent actual-SGD audit passed for source 2000→saved 2090: all 317 Adam
clocks reached 2090, MAIN had 33440 draws (8360 per quarter), and the separate
auxiliary sampler had 360 draws (120 per stratum). Both PCGs were independently
replayed bit-exact. All 128 E7 codec tensors, Flow buffers, RSSM 5152, SR and 2344
Source files remained unchanged; no model parameters were added.

Evidence: [launch](training_runs/renderer_far_geometry_20261112/launch_command.json),
[full continuation plan/proof](training_runs/renderer_far_geometry_20261112/far_wall_phase_plan.json),
[independent actual-SGD audit](training_runs/renderer_far_geometry_20261112/first_far_geometry_update_verification.json),
[real CUDA backward](training_runs/renderer_far_geometry_20261112/gpu_backward_probe.json),
[initial validation](training_runs/renderer_far_geometry_20261112/validation_initial.json),
[validation 2100](training_runs/renderer_far_geometry_20261112/validation/step_002100.json),
[fixed Source depth inventory](training_runs/renderer_objects_walls_20261112/diagnostics/wall_depth_inventory_256.json),
[actual auxiliary-pool smoke](training_runs/renderer_objects_walls_20261112/diagnostics/far_wall_pool_smoke.json).

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
→21×21=441 →21×5=105 →32×1=32 →27×1=27 →16×1=16 →32×4=128
```

This is a request history, not equally complete telemetry for every early run.
Later safe-resume evidence confirms746→747 for27×1 and800→801 for16×1.
Current RSSM uses main-microbatch32, accumulation4, effective128; separate Source auxiliary clips
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
