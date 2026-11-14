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

## H56. Current NPR in the RSSM preview with an immutable v 2 chain

The request to show the latest renderer in “What the model shows” now uses the
completed v 6 NPR at phase 6000 / cumulative 22200, Base 21800, and the same SR best
local 18100 / cumulative 38100. RSSM resumed in the same run from checkpoint 5322
through a preview-only v 2 revision. Training image assets, the pure-XY objective,
all 165 Adam histories, RNG, curriculum and the ten-consecutive-update gate are
retained; architecture and training scope are unchanged. The open game keeps
its own earlier snapshots.

The revision chain retains the original plan (`40d075…`) and v 1 NPR 6600 revision
(`7585e1…`); v 2 is `d7696988…`. The selected NPR file's actual SHA256 is
`a23dab83c6bc9203a31e922046db57747bb5b4890eb13c576354f8577bde795f`.
The preview uses a frozen clone with the current RSSM dynamics, rather than
replacing training or game weights. Source future remains a comparison row,
with no correction of the model's own states or memory.

The independent before/after audit at the same 5322 boundary passed with zero SGD:
all 1321 model tensors,165 Adam states, serialized RNG/PCGs and the entire
`rssm_training` state match exactly, including EMA clocks and N10/H128/streak 0.
All 22 old/new file pins were checked; only operational and preview metadata changed.

Live HTTP and Node VM checks passed at RSSM 5325 / preview 5322: current H128,
128 generated transitions, displayed offsets 1/43/86/128, five rows and cadence 10.
The actual caption identifies NPR 22200 (phase 6000) / SR 38100. The served PNG is
2048×2684 and visibly contains context, Source, Base+RF1, learned SR and bilinear
rows. A deliberately stale other-H snapshot hides its image and displays the
current-H waiting state. Full 450-transition validation remains separate, every 250
updates. This verifies the displayed assets and UI behavior, not gameplay quality
or successful completion of the current dynamics stage.

Evidence: [immutable v 2 revision](training_runs/rssm_loss_gated_drift_20261112/preview_assets_revision_npr6000_v2.json),
[selected frozen assets](training_runs/renderer_far_geometry_20261112/preview_final_006000/preview_assets_npr6000_sr18100.json),
[same-run resume receipt](training_runs/rssm_loss_gated_drift_20261112/preview_revision_resumes/g5322_d76969889ef3/resume.json),
[trainer restore verification](training_runs/rssm_loss_gated_drift_20261112/resume_verification_g5322_loss_gated_drift_v1.json),
[independent same-step audit](training_runs/rssm_loss_gated_drift_20261112/preview_revision_resumes/g5322_d76969889ef3/preview_revision_resume_verification.json),
[preview asset verification](training_runs/rssm_loss_gated_drift_20261112/updated_preview_assets_verification.json),
[immutable first updated preview](training_runs/rssm_loss_gated_drift_20261112/preview_revision_resumes/g5322_d76969889ef3/preview_after.png).

## H57. Shared sparse MoE and simultaneous three-task curriculum

At the user's request, both retained RSSMs now exchange their previous memories
through a shared two-block Transformer, three top-2 feed-forward experts and three
specialized residual heads: player navigation, enemy motion and combat. The
player and enemy memories remain2032+1600=3632 values, updated every tick.
The new module adds3,118,099 parameters; dynamics total48,125,470.

The old pure-XY worker stopped at a complete5352 boundary. A separate immutable
archive and append-only migration preserve all1321 existing tensors, all165 Adam
states/groups and serialized RNG/history; only64 new MoE parameter tensors have a
fresh optimizer group. Zero heads initially reproduce the original function.
Actual GPU checks of two held-out actors over16 own steps passed bitexactly in
both argmax eval and stochastic train modes, including old logits, states,
memories and ending RNG. The latest frozen NPR22200 / SR38100 clone also passed.

The new run `rssm_moe_multitask_20261112` trains all three tasks in each64×4 update.
TOTAL is Player+Enemy+Combat+0.01 role-CE+0.001 load balancing, with fixed explicit
component weights and actual event/axis denominators, no EMA scaling or KL floor.
Auxiliary navigation weight is0.1, late own-prefix weight0.25, TBPTT16. Actual
TRAIN sampler checks passed forH4/8/16:16 spawn+16 positive attacks+16 eligible
negative attacks+16 shooting-replay windows per microbatch; all eight spawn
anchors and requested events fall inside the actual horizon.

The31-stage ladder is4→8→16→32→48→…→432→448→450. A stage advances only after10
consecutive complete optimizer updates with TOTAL<0.25; equality resets the
streak. The next stage begins at0/10, and450 continues. No forced long update is
inserted. Current-H free previews run every10 updates and full450 held-out
validation every250. The dashboard gives this new phase precedence over retained
inactive XY-only metadata and old H128 previews.

Executed first updates:5353 TOTAL3.338475 (Player0.020221 / Enemy0.021814 /
Combat3.292408),5354 TOTAL3.249267. All three specialized heads and both RSSMs had
finite nonzero gradients. The shared experts also had nonzero gradients on the
second update after zero heads opened their gradient path. These two rows do not
establish convergence or improved gameplay; H4 remains at0/10.

Evidence: [migration](training_runs/rssm_moe_multitask_20261112/checkpoints/moe_initial_g5352.pt.fusion.json),
[real GPU comparison](training_runs/rssm_moe_multitask_20261112/fusion_zero_residual_gpu_probe.json),
[immutable phase plan](training_runs/rssm_moe_multitask_20261112/moe_training_plan.json),
[TRAIN sampling](training_runs/rssm_moe_multitask_20261112/sampler_preflight.json),
[metrics](training_runs/rssm_moe_multitask_20261112/metrics.jsonl),
[first-update audit](training_runs/rssm_moe_multitask_20261112/first_updates_verification.json).
Controlled resume at5354 also passed:2075 compared tensors,229 Adam states,
all saved model/optimizer/PCG/CPU+CUDA RNG checks bitexact. Active metadata now
explicitly enables all three tasks; old disabled flags remain in phase history.

## H58. Persistent20M combat RSSM, preserving the current curriculum

The user requested a third RSSM for combat, first15M and then corrected to20M.
The resulting branch has **20,007,020 parameters**, including its condition/readout
networks,268 interaction outputs and a zero512→256 adapter into the existing MoE
combat query. Its categorical memory is h1344+8×32=1600 FP32 values; the packed
player/enemy/combat memory is2032+1600+1600=5232. It updates every tick, including
empty, dead and offscreen enemy cases. Source RGB is used only for observed
initialization/posterior. Current neural P/E readouts remain differentiable inputs;
the shared Transformer reads previous combat memory through the new adapter.

The new branch refines target, attack, spawn tile and spawn anchor logits. Shared
structured_transition remains the only coordinate/HP/event state. Zero task heads
and the zero adapter preserve existing behavior on activation. Its sampling is
isolated with fork_rng so additional categorical draws do not shift the established
P/E random stream. No private mutable RNG clock or extra attention token is added.

The old worker stopped after complete update5437. The separate
`rssm_combat20m_moe_20261112` migration preserved1386 existing tensors,229 Adam
states/groups with mixed ages, serialized RNG, sampler ledgers and the complete
MoE gate (85 completed updates, H4, streak0). Only48 newly added parameter tensors
have a fresh Adam group. The original MoE plan remains pinned; a separate
combat_memory_plan.json records the architecture addition without a horizon reset.

Actual GPU no-SGD checks passed on two genuine Source actors, combat and no-enemy,
over16 steps in argmax and stochastic train modes: all old result fields, old3632
memory prefix and ending CPU/CUDA RNG match exactly. New memory changes every
tick for both actors. Strict latest NPR22200/Base21800/SR38100 loading and native32
→SR128 inference also passed. Eight module, eight backend, five migration and
four continuation-gate CPU tests passed, alongside legacy regressions and83 UI tests.

Training resumed at64×4. Update5438 TOTAL2.189206 trained the new task head and
adapter; at5439 TOTAL2.067064 the deep combat core had finite nonzero gradient
norm0.014764. At5440 TOTAL1.961224, the existing gate had advanced its update
clock85→88 while remaining H4/0-of-10. These observations establish a working
training path, not improved held-out combat or a qualified15-second game.

The live HTTP/VM/PNG audit passed23 checks: measured20,007,020 parameters,
5232 memory, three-RSSM phase label, fresh five-row H4 preview with NPR22200/SR38100,
correct TOTAL arithmetic and stale-horizon image hiding. Validation remains every
250 MoE updates, preview every10, with the same ten-consecutive-TOTAL<0.25 gate.

Evidence: [architecture migration](training_runs/rssm_combat20m_moe_20261112/checkpoints/combat20m_initial.pt.combat-memory.json),
[GPU probe](training_runs/rssm_combat20m_moe_20261112/combat_memory_gpu_probe.json),
[continuation plan](training_runs/rssm_combat20m_moe_20261112/combat_memory_plan.json),
[restore](training_runs/rssm_combat20m_moe_20261112/resume_verification.json),
[live UI audit](training_runs/rssm_combat20m_moe_20261112/diagnostics/live_combat_ui_g5440_20261112T175746Z/proof.json).

## H59. Restore late respawn supervision by aligning death-clock arithmetic

The combat-loss audit on5438–5453 found spawn contributed68.9% of the mean and
80.9% of the observed variation. A reproducible numerical boundary censored late
labels: normalized death age4.9444441795 produced Torch3.0s but Source2.999999841s.
Torch revived the slot before the conditional Source query accepted a birth.
Across4096 selected auxiliary windows,1043 recorded births yielded only264 labels,
all at offset1;779 later labels disappeared.

The authorized fix introduces a shared death/respawn phase advance at30Hz. Only
float32 representation error near a simulation phase is recovered at STEP time.
Literal Game.set_state_vector, strict canonical2e-7 checking, numerical state
inputs and own h/z remain untouched. Fire/attack/flash/hurt/global-time clocks are
unchanged. Shared death readiness now covers structured_transition, learned and
hierarchical spawn decoders, spawn-render loss and runtime invariants. Generic dt
and genuinely off-grid states keep their documented fallback. Normal Source
128-step replay remains bitexact in events, serialized states and32px RGB.

Independent replay of exactly the same panel restored **1043/1043 labels**:
offsets264/267/240/272, with exact birth times and anchor labels. All eight anchor
cohorts now have128–135 events instead of25–41. Four coverage regressions, seven
clock tests, Source interop tests and existing structured/spawn regressions passed.
No new Source corpus was needed; existing data files were unchanged.

The old worker stopped at complete5476. Its entire checkpoint was copied
bitexactly into `rssm_respawn_timing_20261112`:1434 tensors,277 Adam states,
serialized RNG, sampler ledgers and both existing phase clocks (MoE124,
Combat39,H4/streak0). An immutable `--respawn-timing-plan` pins the corrected
rule files; architecture/parameter counts,64×4, the ten-consecutive-TOTAL<0.25
gate, preview10 and validation250 remain unchanged.

The disposable GPU probe passed14 checks without SGD: real B64×H4 backward for
contexts1/2 had finite nonzero P/E/Combat gradients; all550 legal conditional
queries agreed on respawn, with positive labels at every offset.26 geometry
queries were honestly excluded. Two own450 rollouts kept finite5232D memory and
revived exactly at tick90. Strict NPR22200/SR38100 inference passed. This is not a
held-out gameplay-quality pass; invalid own geometry remains a separate issue.

Actual update5477 produced61 spawn labels at offsets14/21/16/10, with7–8 per
anchor. At5479 the count was57 at15/16/16/10. Logs now record seven flat Combat
contributions; the dashboard shows aggregate offset/anchor counts from all four
microbatches and timing-revision identity.84 UI tests and29
live HTTP/VM/PNG checks passed. Loss before/after this coverage correction uses
different event sets; short-term raw changes do not establish better quality.

Evidence: [original diagnosis](training_runs/rssm_combat20m_moe_20261112/diagnostics/loss_audit_20261112/respawn_clock_probe.json),
[same-panel coverage](training_runs/rssm_combat20m_moe_20261112/diagnostics/loss_audit_20261112/respawn_revision_coverage.json),
[timing plan](training_runs/rssm_respawn_timing_20261112/respawn_timing_plan.json),
[GPU probe](training_runs/rssm_respawn_timing_20261112/diagnostics/respawn_timing_gpu_probe.json),
[restore](training_runs/rssm_respawn_timing_20261112/resume_verification.json),
[live UI](training_runs/rssm_respawn_timing_20261112/diagnostics/live_respawn_ui_g5479_20261112T184303Z/proof.json).

## H60. Loss streak OR stage budget without resetting training

The user added a per-stage time budget to the existing three-task curriculum.
The separate immutable `rssm_moe_loss_or_stage_budget_v2` plan advances after
**10 consecutive complete optimizer updates with strict TOTAL loss <0.25 OR
`50 + 15*N` completed updates at that stage**, with zero-based N. The original
objective plan, loss weights,64×4 batch, Source physics and frozen image assets
remain unchanged. Final H450 continues; a budget advance does not qualify gameplay.

The old worker stopped at5542. Its1434 model tensors,277 Adam states/groups and
serialized RNG were copied and restored exactly into `rssm_moe_budget_20261112`.
The existing190 H4 updates were retained: activation advanced exactly once at
the saved5542 boundary to H8, N=1, cap65. No historical stage count was reset or
invented. The new worker actually resumed; at5545 it had completed three H8
updates. Monitoring remains **http://127.0.0.1:8765/**. Full450 validation is
separate from stage progression; no new gameplay-quality pass is claimed.

Evidence: [budget plan](training_runs/rssm_moe_budget_20261112/stage_budget_plan.json),
[bitexact source copy](training_runs/rssm_moe_budget_20261112/preparation.json),
[model/Adam/RNG restoration](training_runs/rssm_moe_budget_20261112/resume_verification.json),
[actual launch](training_runs/rssm_moe_budget_20261112/launch_command.json),
[completed-update log](training_runs/rssm_moe_budget_20261112/metrics.jsonl).

## H61. RGB-only semantic SR on actual neural-renderer pairs

The user requested learned restoration of blurry enemies, in parallel with
dynamics. The completed dataset contains **9216 pairs:8192 TRAIN/1024 VAL**,
with equal near/mid/far/guard cohorts. Each input is actual frozen NPR32 using
world5510/NPR22200/E7; the target is true same-pose Source128, never resized LR.
Source masks supervise loss only. Source seeds/render conditions are held out;
the cache audit records four identical cross-split NPR inputs, so NPR-byte
disjointness is not claimed.

On the same1024-frame NPR panel, SR40100 improved over SR38100 in every distance
band and was selected as parent. Its full matching checkpoint retained74 Adam
states at step2000. The new **1,667,368-parameter** Semantic EDSR appends149,797
parameters for RGB-only mask/presence and a softly gated HR residual; the zero
RGB head preserves the parent function initially. No atlas or Source mask is
inserted at inference. A real GPU audit verified matching parent weights/moments,
zero-append RGB equality and two disposable SGD steps without changing sources.

The actual SR worker runs in `sr_semantic_npr_20261112/training`, alongside
RSSM. Batch8 uses6 NPR inputs and2 clean/codec guards. **Preview15, full validation100**
are separate; live checks observed image updates615→630 while full metrics retained
their own validation step. Progress: **http://127.0.0.1:8779/**.

| Actual NPR foreground MAE, fixed VAL1024 | SR38100 | Parent40100 | Semantic step500 |
|---|---:|---:|---:|
| Near,256 frames | .12334 | .12152 | .11808 |
| Mid,256 frames | .13604 | .13205 | .13021 |
| Far,256 frames | .14033 | .13568 | .13341 |

At VAL500, NPR mean SSIM was.779728 and MAE.032607. The auxiliary presence head
still had false-positive rate1.0 on178 negative frames. Enemies remained blurry;
RGB improvement is not demonstrated recognition. All quality goals remained
false. The game, RSSM preview and active SR38100 were not replaced automatically.

Evidence: [Source-pair audit](training_runs/sr_semantic_npr_20261112/source_pairs_verification.json),
[actual NPR cache audit](training_runs/sr_semantic_npr_20261112/npr_cache_verification.json),
[parent comparison](training_runs/sr_semantic_npr_20261112/preflight/baseline_comparison.json),
[GPU/migration proof](training_runs/sr_semantic_npr_20261112/preflight/semantic_sr_gpu_preflight.json),
[training plan](training_runs/sr_semantic_npr_20261112/training/training_plan.json),
[actual launch](training_runs/sr_semantic_npr_20261112/training/launch_command.json),
[VAL500](training_runs/sr_semantic_npr_20261112/training/validation/step_000500.json),
[live preview/UI proof](training_runs/sr_semantic_npr_20261112/training/diagnostics/live_sr_ui_s600_20261112T195546Z/proof.json).

## H62. Restore the SR training chart and measure validation loss

The SR chart rejected saved rows because they used `objective_protocol` while
the generic filter required `loss_protocol`. A strict semantic-SR adapter now
accepts the matching rows, and the server samples the whole training history
into at most300 points while retaining actual validation measurements. From
step1900, validation computes the same weighted semantic objective on1024 held-out
NPR inputs; training mixes6 NPR and2 clean/codec inputs. RGB MAE is separate.
The controlled restart at1800 retained all102 model tensors and102 Adam states,
RNG and sampler. Live step1955 showed train0.23336 and validation0.24891;
preview15/validation100 and parallel dynamics continued. Validation loss for
earlier steps is unavailable and is not inferred from image metrics.
Evidence: [restart](training_runs/sr_semantic_npr_20261112/training/diagnostics/loss_chart_restart/resume_verification.json),
[live chart](training_runs/sr_semantic_npr_20261112/training/diagnostics/live_loss_chart_s1955/proof.json).

## H63. Learned enemy decoder with predicted RSSM proposals

The user authorized a learned decoder conditioned on a predicted RSSM hypothesis.
After parent semantic SR7500/cumulative47600 stopped, all102 old model tensors,
Adam states and recorded RNG were retained. The new1,957,382-parameter model adds
290,014 parameters/43 fresh tensors: learned canonical RGBA32 → bbox warp → HR
visible mask, with RGB-only ROI confirmation before composition. No atlas or
Source state/mask is an inference input. Zero gain preserves the parent image;
unknown/invalid hypotheses retain RGB-only fallback. The GPU preflight passed,
including three disposable SGD steps without modifying the initial checkpoint.

The first worker ran in `sr_enemy_decoder_20261112/training`, alongside dynamics,
at **http://127.0.0.1:8779/** with preview15/full validation100. Frozen world5570
(`b2480e27…`) supplied state20+validbit sidecars for9216 pairs:6736 valid one-step
past-frame priors and2480 unknowns. This bootstrap is not autonomous450 validation.
The new sampler balances actual HR-visible/empty cases50:50;25% of predicted hints
are permuted during training, while Source masks remain truthful loss labels.
Runtime/model/data/integration tests passed. No quality or automatic deployment
is claimed. It later stopped at400: VAL SSIM fell.7826→.1609, and a real diagnostic
found old-EDSR gradient norm1.17e15. Its checkpoint/archive remain; H64 supersedes
this training attempt. Evidence: [gradient diagnosis](training_runs/sr_enemy_decoder_20261112/diagnostics/gradient_instability_probe.json),
[GPU proof](training_runs/sr_enemy_decoder_20261112/diagnostics/gpu_preflight.json),
[hypothesis audit](training_runs/sr_enemy_decoder_20261112/hypotheses_verification.json),
[actual launch](training_runs/sr_enemy_decoder_20261112/training/launch_command.json).

## H64. Frozen-parent SR and predicted hit/death animation

After the failed400-update run, the new stable profile starts from the good
semantic7500/cumulative47600 parent. All102 old weights and Adam states/RNG are
retained; old RGB and its shared classifier gates are frozen. Only47 new tensors
train:290,854 parameters,1,958,222 total. A real GPU three-step proof kept every
parent tensor/moment unchanged. The canonical decoder now receives seven animation
classes from predicted RSSM state, with independent RGB visual-phase CE. Source
animation labels supervise loss only; conflicting phases do not supervise canonical
RGBA. No atlas is used.

The actual worker is `sr_enemy_decoder_stable_20261112/training`, on
**http://127.0.0.1:8779/**, with preview15/validation100 alongside unchanged dynamics.
Sampling is50:50 actual visible/empty, balancing available phases within all three
distances. Existing Source TRAIN hurt/death-start/death-late counts1323/162/355
and VAL184/20/40 needed no new collection. At VAL100, SSIM.782579/MAE.0326466 matched
the parent; confirmation recall/FPR0/0 does not establish recognition. No quality
or deployment is claimed. Evidence: [actual GPU preservation proof](training_runs/sr_enemy_decoder_stable_20261112/diagnostics/gpu_preflight.json),
[launch](training_runs/sr_enemy_decoder_stable_20261112/training/launch_command.json),
[VAL100](training_runs/sr_enemy_decoder_stable_20261112/training/validation/step_000100.json).

## H65. Fifty updates per stage without resetting training

The active dynamics worker60312 is `rssm_moe_cap50_20261113`, continued from the
complete5596 checkpoint with1434 tensors,277 Adam states and RNG preserved.
The v3 policy is **10 consecutive strict TOTAL loss<0.25 OR50 updates per stage**,
including final H450; its budget stop is `completed_budget`, not a quality pass.
Historical H4=190/H8=54 counts remain. Activation advanced once to H16/0 without
SGD or a clock reset. The old `50+15*N` policy remains in H60; the interim100
request was never activated.

Original objective,64×4 batch, Source physics, preview10 and full450 validation250
are unchanged. Stable SR67708 continues separately; no quality or deployment is
claimed. Evidence: [cap50 plan](training_runs/rssm_moe_cap50_20261113/stage_budget_plan.json),
[activation preview](training_runs/rssm_moe_cap50_20261113/budget_activation_preview.json),
[exact model/Adam/RNG restore](training_runs/rssm_moe_cap50_20261113/resume_verification.json),
[actual launch](training_runs/rssm_moe_cap50_20261113/launch_command.json).

## H66. RGB-first enemy SR with optional RSSM hints

Worker52968 initially ran `sr_visual_first_20261113/training`, on8779, alongside cap50
dynamics. Confident RGB chooses geometry, appearance and animation without an
RSSM veto; hints assist only uncertain RGB-confirmed cases. The model has
1,967,824 parameters,300,456 trainable. Stable5500/cumulative53100 supplied all149
Adam states/RNG and the active44,000-draw sampler;9 new tensors add9,602 parameters.
Three GPU updates preserved the102 frozen parent tensors. Source labels are loss
targets only; no atlas is used.

Hint training is50% dropped/25% wrong/25% retained, with75% actual NPR inputs and25%
guards. Six-row with/no-hint previews run every15; validation every100 uses all1024
held-out frames (845 visible/179 empty) in each hint mode. Initial no-hint detections0
establish no quality; the open game's immutable weights remain unchanged. Evidence:
[migration](training_runs/sr_visual_first_20261113/inputs/visual_first_initial.pt.visual-first.json),
[GPU preflight](training_runs/sr_visual_first_20261113/diagnostics/gpu_preflight.json),
[launch](training_runs/sr_visual_first_20261113/training/launch_command.json),
[initial validation](training_runs/sr_visual_first_20261113/training/validation_initial.json).

## H67. Twenty-five updates or five TOTAL-loss passes per stage

The user replaced cap50 with a maximum of25 completed optimizer updates per
horizon, or an earlier advance after five consecutive strict TOTAL losses<0.5.
`rssm_moe_cap25_20261113` continues the same31-stage H4→H450 objective and64×4
batch. Loss≥0.5 resets the new streak. The v4 policy is pinned separately from
the unchanged MoE objective; older policies and their losses remain history.

Activation at global5741 changed H48/45→H64/0 with **zero SGD updates**. The
389-update objective clock,1434 model tensors,277 Adam states and serialized
CPU/CUDA RNG were preserved. The restoration receipt compares2267 tensors
across the saved model, optimizer and RNG payload. A fresh streak starts at0;
previous loss values are not retrospectively judged by the new threshold.
Final H450 stops at25 updates (`completed_budget`) or its five-loss streak
(`completed_loss_gate`); neither is a450-transition gameplay-quality pass.
Source labels remain targets only; persistent h/z and the frozen renderer are
unchanged by this policy revision.

The live HTTP/served-JS/PNG check at5741 passed22 checks: H64/0 over25,
the389-update clock, prior H4=190/H8=54 history, a finite five-row H64 preview
and pinned NPR22200/SR38100. Old-H48 images and stale budget pins are hidden.
The first new SGD committed5742: H64/1, objective390, policy1. TOTAL loss was
1.2074103928 and the finite world gradient norm17.15399; this does not pass the
new loss threshold. The post-update UI check also passed22 checks.
The full copied source checkpoint is available at
`checkpoints/initial_before_cap25.pt`; later checkpoints continue training.
Evidence: [immutable policy](training_runs/rssm_moe_cap25_20261113/stage_budget_plan.json),
[activation](training_runs/rssm_moe_cap25_20261113/budget_activation_preview.json),
[bitexact restore](training_runs/rssm_moe_cap25_20261113/resume_verification.json),
[live UI](training_runs/rssm_moe_cap25_20261113/diagnostics/live_cap25_ui_g5741_20261113T045920Z/proof.json),
[first new update](training_runs/rssm_moe_cap25_20261113/diagnostics/live_cap25_ui_g5742_20261113T050242Z/proof.json),
[launch](training_runs/rssm_moe_cap25_20261113/launch_command.json).

## H68. Raw RGBA restoration before gradual blending

The fixed near/mid/far decoder figures at visual-first update43100 (cumulative
96200) showed diffuse raw RGB, a coarse alpha silhouette and visual gain0.0136658.
The explicit forced-gain1 diagnostic did not create sharp enemy detail. These
are three fixed validation cases, not a complete quality evaluation:
[raw RGB/alpha](training_runs/sr_visual_first_20261113/training/diagnostics/decoder_raw_s43100/raw_decoder.png),
[actual and forced outputs](training_runs/sr_visual_first_20261113/training/diagnostics/decoder_raw_s43100/full_frames.png),
[snapshot/case metadata](training_runs/sr_visual_first_20261113/training/diagnostics/decoder_raw_s43100/metadata.json).

An explicit `decoder_restoration_sr_v3` phase was launched from that checkpoint
in `sr_decoder_restoration_20261113/training`. No parameter tensor is added:
the model remains158 tensors,1,967,824 parameters/300,456 trainable, with102 frozen
parent tensors. The opt-in config records `enemy_decoder_restoration=True` and
`restoration_blend_scale=0`. Existing visual-first and guided profiles remain separate.

Raw RGB/alpha, SSIM, edges and raw composites receive losses before learned
display gains and routing. Label crops and layer warps use the same detached
actual predicted box; a tight Source visible bbox is not stretched across a
larger prediction. Source foreground is premultiplied by its visible mask before
area pooling to32×32, preventing wall/sky colors from contaminating partly
covered foreground color targets. Source RGB, background and masks remain labels
only. Detection and optional causal RSSM-hint supervision continue.

The saved schedule keeps the layer contribution at0 for at least200 optimizer
updates. On each complete fixed validation100, raw foreground MAE and raw edge
MAE must both strictly improve over the initial report; empty/background errors
must not worsen. Until this relative gate passes, standalone training continues.
After passing,400 subsequent optimizer updates linearly raise the saved blend
scale0→1. The checkpoint retains phase, scale, counters, initial-report identity
and decisions; previews remain every15. This blend gate does not establish the
overall SR thresholds or autonomous gameplay quality.

At preparation,38 focused CPU tests passed. A7.016-second GPU preflight performed
three disposable optimizer updates with finite RGB/alpha gradients, while all102
frozen parent tensors/Adam states remained unchanged. All158 source model tensors
and158 Adam states/options/order, serialized RNG and388,800 active sampler draws
were preserved; no parameter was added. These disposable updates are not
production training progress. Worker67500 launched with scale0, previews15 and
validation100; automatic game deployment remains disabled. New image-quality
improvement is not yet claimed. Evidence:
[migration](training_runs/sr_decoder_restoration_20261113/inputs/restoration_initial.pt.restoration.json),
[GPU preflight](training_runs/sr_decoder_restoration_20261113/diagnostics/gpu_preflight.json),
[actual launch](training_runs/sr_decoder_restoration_20261113/training/launch_command.json).

The actual HTTP/served-JS/PNG check at local200 with validation100 passed49 checks:
standalone phase, saved scale0,200 train points/two real validation points,
six full preview rows plus three ROI rows, and the separate raw-quality card.
The initial-report SHA, panel and data identity match. On the same1024-frame
panel, raw foreground MAE changed0.130125899→0.129576445 (−0.42%), raw foreground
edge0.134333679→0.132599063 (−1.29%), empty0.003311620→0.000844258 and background
0.003475664→0.002298815. These are before-gain raw metrics; foreground/edge use845
Source-visible frames, empty179 and background1024. The report at100 did not meet
the minimum200-update blend condition; the captured state still had scale0.
Final RGB MAE remained0.032646591 across hint modes, so no displayed-image quality
improvement is claimed. Evidence:
[live UI](training_runs/sr_decoder_restoration_20261113/training/diagnostics/live_restoration_s200_20261113T051104Z/proof.json),
[same-panel raw comparison](training_runs/sr_decoder_restoration_20261113/training/diagnostics/live_restoration_s200_20261113T051104Z/raw_delta_at100.json).
Implementation:
[raw objectives](turkeyworld/decoder_restoration_objectives.py),
[persistent schedule](turkeyworld/decoder_restoration_protocol.py).

## H69. Matched ROI PatchGAN and NoGAN control

The user authorized a small conditional enemy-ROI PatchGAN with a fair trained
NoGAN control. The launched `enemy_roi_patchgan_ab_v1` phase forks restoration
update1830/cumulative98030 into identical generators with the same158 model/Adam
states, RNG and sampler. NoGAN is a learning control, not a frozen ancestor.
One shared query supplies row IDs, domain mix, NPR input and optional causal
RSSM hint to both arms; a rolling batch-input ledger and equal generator-step
counts make this comparison auditable. Both saved restoration blends are1.

The separate spectral-normalized discriminator sees a detached NPR ROI condition
and real/fake RGB+alpha on the same crop/canvas. Fake raw RGBA uses the actual
predicted box before gain or routing. Source RGB/masks define real labels only;
Source future is never a generator input. Existing RGB/alpha/geometry/edge
reconstruction losses remain active. GAN weight is0 during100 warmup updates,
then ramps over400 updates to at most0.01, with an additional0.1 gradient-ratio
limit measured on the shared RGB/alpha readout. Invalid ROI batches may skip a
discriminator step; their counts must not be invented as successful D updates.

Evaluation uses the same1024 held-out frames for both arms, including near/mid/far,
foreground, silhouette and empty-scene supports. The existing fixed panel also
contains138 proven adjacent cached one-step causal NPR pairs. Temporal errors
compare the predicted image change with the true Source change; they are not an
autonomous neural-game video result. Missing pairs produce unknown metrics,
not zero error. Reconstruction validation excludes GAN losses; generator total,
adversarial, feature matching and D losses remain separate diagnostics.

The dashboard requires matching plan/source/panel/data pins, equal arm
steps and the appropriate batch ledger for the report's step. Four preview rows
show Source128 / NPR32 / trained NoGAN / GAN on the same fixed cases. An older
valid report may have an older ledger than current training; mismatched arm
steps or panels are rejected.122 Node UI tests and six evaluator CPU tests
passed. The7.38-second GPU preflight preserved all158 generator model/Adam states
per arm, all parent RNG and403,440 active sampler draws from cumulative98030.
The zero-weight probe produced bitexact generator weights/Adam in both arms;
three forced0.01-weight disposable probes reached RGB/alpha with monitored gradient
ratios below0.1. All102 frozen parent tensors remained unchanged in both arms.
The206,889-parameter discriminator has a separate fresh Adam. These probes were
discarded and are not production progress.

Worker74080 launched `sr_patchgan_ab_20261113/training` on8779. The live
HTTP/served-JS/PNG check at paired200/validation100 passed33 checks: GAN/NoGAN/D
steps200, scheduled weight0.0025/effective0.0008850393, four-row preview195,
1024 held-out images and138 temporal pairs. Both reconstruction curves were
real and excluded GAN total/control losses. The prior warmup validation100 had
equal arms; that equality is not evidence for a GAN advantage.

The first active validation200 gave mixed observations on the same panel:

| VAL200 metric | GAN | Matched NoGAN |
|---|---:|---:|
| ROI crop SSIM ↑ | 0.2735549345 | 0.2732015178 |
| Image SSIM ↑ | 0.773789014 | 0.773723594 |
| Silhouette IoU ↑ | 0.432603435 | 0.434478600 |
| Source sharpness error ↓ | 0.11131582 | 0.11146029 |
| Empty layer false-positive rate ↓ | 9/179 | 9/179 |
| Temporal FG delta MAE ↓,137 foreground pairs | 0.134167705 | 0.134144908 |

Image/ROI SSIM increased slightly, while silhouette and temporal foreground
results worsened slightly. This snapshot establishes no overall GAN quality or
autonomous-gameplay advantage. No automatic game promotion is enabled. Evidence:
[full-state twin initialization](training_runs/sr_patchgan_ab_20261113/inputs/initial.pt.patchgan.json),
[GPU preflight](training_runs/sr_patchgan_ab_20261113/diagnostics/gpu_preflight.json),
[launch](training_runs/sr_patchgan_ab_20261113/training/launch_command.json),
[live A/B page](training_runs/sr_patchgan_ab_20261113/training/diagnostics/live_patchgan_s200_20261113T055217Z/proof.json),
[active validation200](training_runs/sr_patchgan_ab_20261113/training/validation/step_000200.json).
Implementation:
[conditional ROI discriminator](turkeyworld/sr_enemy_patchgan.py),
[matched continuation protocol](turkeyworld/sr_patchgan_protocol.py),
[paired image/temporal evaluation](turkeyworld/sr_patchgan_validation.py).

## H70. Fixed-atlas enemy appearance with learned geometry and visibility

The user authorized an explicit hybrid after the learned RGBA remained diffuse.
The running `atlas_enemy_hybrid_sr_v1` profile uses
`sr_atlas_enemy_20261113/training` on8779. Enemy appearance comes from the fixed atlas;
the NN learns full sprite geometry, animation phase, occlusion and binary
visibility. This is not a claim of learned enemy-pixel generation. The learned
parent SR supplies the background and is separately frozen as a reference;
there is no matched NoGAN or discriminator training in this phase.

The generator has1,967,873 parameters,70,195 trainable. The158 inherited parameter
tensors/Adam states are retained; only a49-parameter RGB HP0 head adds two parameter
tensors. Unused learned paint modules remain frozen. The RGBA asset is embedded
as a persistent uint8 buffer `[7,4,444,444]` plus native frame sizes. The original
`assets/turkey_atlas.png` is1774×887, SHA
`56bec622ff615462971de4d68c982a319c5928f069d6047b93eafd0c210b7ac6`.
Its four-by-two Source cuts use rounded boundaries, so native frames vary443/444
pixels; frames0–6 are used. Checkpoints pin the file and embedded-pixel hashes.
Strict loading uses the embedded buffers without an external asset or game engine.

The full box remains unclipped so offscreen sprites keep their UV coordinates;
it is not the tight bbox of visible mask pixels. Native atlas pixels are warped
directly to128, with a hard phase choice and hard forward visibility. Straight-
through gradients train geometry, phase and mask. Source full boxes, phases,
visible masks and RGB are labels only. Runtime uses actual NPR32 and an optional
current predicted RSSM hint; it never asks Source for a future pose or render.

The new dashboard separates full-bbox IoU/L1, phase accuracy, alpha/visibility
IoU, empty composite false positives and image/temporal metrics with their support
counts. Four preview rows are Source / NPR / frozen learned SR / Atlas SR. The
new NN content-loss graph excludes the frozen reference's different objective
and prior GAN losses.128 Node tests,20 local-server tests and32 focused/legacy CPU
runtime tests passed. The server retains all actual weighted validation points
and uniformly samples at most300 train points; MAE/reference/GAN losses cannot
stand in for atlas validation loss.

Worker40076 launched from stopped PatchGAN2900/cumulative100930, SHA
`27405e2aac6323d6b7ebaa833ef3d52b81e56b3a4eeeb223e3febdb9b3497119`.
Its7.75-second real GPU preflight verified158 inherited tensors/Adam states,
all RNG/sampler state, three disposable optimizer steps, unchanged120 frozen
tensors/Adam states and embedded atlas buffers. Only the new two-tensor HP0 head
starts with fresh Adam. The production source and initial files remained unchanged.

The archived live HTTP/served-JavaScript/PNG check passed37/37 at update420,
validation400 and preview420. It verified1024 held-out images,138 adjacent cached
NPR pairs, exact plan/source/data/panel pins, both measured loss curves and the
four-row1024×1120 image. Preview cadence is15; full validation cadence is100.
Initial→VAL100 weighted loss was0.817255→0.716567, bbox L1 error10.2171→6.7076px,
visibility IoU0.345981→0.394582 and empty composite FPR0.078212→0.050279.
Phase accuracy worsened0.272189→0.197633. The fixed sprite is sharp, but placement
and occlusion remain imperfect; these observations do not qualify the hybrid.
The existing game and seven pinned physics files remain untouched. Evidence:
[GPU preflight](training_runs/sr_atlas_enemy_20261113/diagnostics/gpu_preflight.json),
[launch](training_runs/sr_atlas_enemy_20261113/training/launch_command.json),
[live page proof](training_runs/sr_atlas_enemy_20261113/training/diagnostics/live_atlas_s420_20261113T065824Z/proof.json),
[VAL100](training_runs/sr_atlas_enemy_20261113/training/validation/step_000100.json).
Implementation:
[embedded-atlas model](turkeyworld/atlas_enemy_sr.py),
[full-geometry/mask objectives](turkeyworld/atlas_enemy_objectives.py),
[hybrid validation](turkeyworld/atlas_enemy_validation.py).

## H71. All-depth wall geometry refinement with SR stopped

The user stopped atlas SR at saved update5175 and requested NPR wall/corner
repair. The stopped atlas archive remains on8779. Cap25 dynamics and the open
game are independent; this renderer phase does not replace their pinned images.

Worker12752 runs `renderer_wall_repair_20261113`, continuing the complete v6
renderer from counter6000 to8000 under `one_enemy_all_depth_wall_geometry_renderer_v7`.
Its preserved restart boundary is6001.
The source SHA is `0f20f72e9c9353576cbbf2cde299d4acbf7c83170458f7d8466dc1d49d60ddbd`.
No architecture or model parameters are added. All317 Adam states, Base/RF1,
RNG, MAIN sampler and old far-wall sampler are retained; the new geometry
sampler has its own TRAIN-only pool and draw clock. RSSM5152, map, E7 codec and
SR remain frozen, so same-Source-pose comparisons isolate rendering errors.

Each update keeps original MAIN16 and far-wall aux4 with weight0.25, and adds
geometry aux8 with weight1. Its fixed8192-candidate pool balances near corners,
mid corners, far thin walls and other far walls. All five disjoint depth bands
(0–2,2–5,5–8,8–12,12+) receive present-only normalized wall RGB0.25, boundary0.3,
edge0.2, corner0.2, thin-face0.2, thin-height0.1 and cross-boundary contrast0.2.
Source map/poses/masks/RGB define labels only; no correction is returned to RSSM.

The additional independent128-image geometry VAL panel is drawn from4096
held-out candidates. Main256, native-area256 and distance192 guards remain.
Validation cadence is100, native32 preview10. The page at8781 separates
Base/RF1/codec empirical errors and actual support by depth on the main and
geometry panels; absent regions are unmeasured. It does not treat codec as a
proven quality limit or image improvements as autonomous gameplay qualification.

The discarded5.2-second GPU backward probe passed with Base/RF1 gradient norms
3.9795/1.4677, all317 Adam states unchanged and no production update in that probe.
The worker has launched; preparing status and a completed probe alone do not
establish training or quality improvement. Actual TRAIN inventory found2176 near,2176 mid,2176 far-thin
and195 other-far candidates; the independent128 VAL has32 per stratum with camera
quadrants balanced and no TRAIN overlap. No new Source collection was needed.
The new page runs on8781;8779 remains the stopped SR archive. Evidence:
[plan](training_runs/renderer_wall_repair_20261113/launch_plan.json),
[discarded probe](training_runs/renderer_wall_repair_20261113/gpu_backward_probe.json).

The initial worker30140 failed while generating the first lightweight wall
preview: an NPC-oriented evaluator expected `foreground_mae` on all-empty wall
queries. Durable6001 had already saved all317 Adam states, MAIN96016/far16004/
geometry8 draws. Reported but unsaved6002–6009 metrics were archived and removed
from the active curve; they are not claimed as retained updates. The resumed
worker12752 loads6001 and a dedicated all-empty-safe renderer preview, without
resetting the phase or optimizer. The new four columns are fixed near corners /
mid corners / far thin / other far, and status/preview metadata pin their panel,
Source indices and actual step. Old NPC preview is hidden until those pins match.
The correction passed23 geometry CPU tests, including real Source/PNG cases;
the existing focused regression suite passed51, and the UI passed132 Node tests.
The failure and the original attempt remain in
[diagnostics](training_runs/renderer_wall_repair_20261113/diagnostics/first_preview_failure).

Actual HTTP/served-JavaScript VM/PNG proof passed34/34 at6191, VAL6100 and
geometry preview6190. It verified the317-state adoption proof, actual auxiliary
draws, both Source panels,30 model/depth rows, exact geometry columns and rejection
of stale NPC images. The640×740 PNG was inspected: geometry is still visibly
blurred and displaced. On the fixed main256 panel RF SSIM0.743642→0.753754 and
MAE0.0339271→0.0335728 improved, but geometry-preservation and overall release
gates remain false. This is early rendering evidence, not repaired-wall or
dynamic/gameplay qualification. See [live proof](training_runs/renderer_wall_repair_20261113/diagnostics/live_geometry_g6191_20261113T080253Z/proof.json)
and [VAL6100](training_runs/renderer_wall_repair_20261113/validation/step_006100.json).

## H72. Atlas SR in dynamics preview and separate navigation diagnosis

The user requested the new upscaler in the active dynamics display. The stopped
Atlas5175/cumulative106105 checkpoint was exported without optimizer/RNG/sampler
state, retaining its strict embedded atlas and lineage. Its source SHA is
`3aebf58b1122f04c89a1e25f077b600d86673d717ea137b46a6279a26e105d4a`;
the immutable play/preview export SHA is
`6e90c520acdfd22162d5ca6b068fe9ebb54bc55527727fbe7d291fb455baf550`.
SR training remains stopped at5175 on8779. V7 NPR wall refinement continues
separately on8781; the display retains NPR22200/phase6000, not the new candidate.

An explicit SR-only v2 descendant pins the new export and preserves the NPR
entry exactly. The preceding SR38100 plan/revision remain in the verified chain.
Only `--preview-assets-revision-plan` changed when dynamics resumed as worker8872
at the same5797 boundary. Independent before/after checkpoint verification found
all world tensors,277 Adam states/options, RNG, source samplers, immutable training
protocol and MoE gate identical before any SGD. H96, MoE445, stage index7 and
stage-updates6 were preserved. No architecture/loss/curriculum reset occurred.

All12 preview-assets CPU tests passed, including strict self-contained Atlas
loading with Source rendering and external atlas loading forbidden. The existing
SR preview call supplies only generated NPR RGB; optional hypothesis is None,
explicitly `atlas_hypothesis_source='none_RGB_only'` in the selected SR metadata.
No future Source pose/mask/image is an SR input or world correction. Actual8765
HTTP/served-JavaScript VM/PNG proof passed23/23 at5797: H96/context2, offsets
1/33/64/96, five rows, NPR22200 and SR106105. The open game was not replaced,
and this display update does not establish quality qualification. Evidence:
[export](training_runs/rssm_moe_cap25_20261113/preview_assets/atlas_sr_106105_export.json),
[revision](training_runs/rssm_moe_cap25_20261113/preview_assets_revision_atlas_sr106105_v2.json),
[same-step restoration](training_runs/rssm_moe_cap25_20261113/atlas_sr_preview_resume_verification.json),
[actual page proof](training_runs/rssm_moe_cap25_20261113/diagnostics/live_atlas_preview_g5797_20261113T081002Z/proof.json).

A separate read-only navigation audit compared the same16 no-enemy450-step
windows at5602 and5797. Mean final player XY improved0.5304→0.4446 tiles, while
wall-interior state steps increased611→634; both remain unqualified. This does
not prove a whole-navigation regression or establish forgetting as its cause.
An additional comparison with5194 uses the same128 short/context1 cases and
the same16×450 windows: false stops20/168→40/168, missed blocks5/40→4/40,
final XY0.3421→0.4446 tiles, states inside walls440→634. Partial navigation
regression relative to5194 is therefore measured. Missing no-enemy replay is
confirmed in code; its causal role remains a hypothesis. The diagnostic report
now retains the5194 source report and matching panel fingerprints.
Combat-focused sampling still requires independent no-enemy checks.
For the shown H96 case, offset64 had0.6284-tile player error; replacing only the
player pose with Source for an offline renderer counterfactual raised SSIM
0.4416→0.8628 with unchanged neural features. At offset33, even the full Source
pose produced SSIM0.5728, exposing a separate NPR rendering limitation. Zeroing
the128-dimensional features did not consistently repair the images. These
counterfactuals are diagnostics only and never feed the training/runtime state.
See [report](training_runs/diagnostics/navigation_latest_20261113_g5797/diagnostic_report.json)
and [pose/feature panel](training_runs/diagnostics/navigation_latest_20261113_g5797/preview_pose_feature_counterfactual.png).

## H73. Three-way MAIN replay and twenty updates per stage

The user required at least15% navigation and25% pure NPC movement in the
effective MAIN batch. Each64-example micro now runs38 combat,16 NPC motion and10
no-enemy navigation actors. Four micros give152/64/40 of256:59.375%/25%/15.625%.
Combat AUX64/micro and late-prefix8/micro remain separate (256/32 actors per
optimizer update). They are not included in the MAIN-share denominator.

The original combat draw clock is preserved:64 source indices are reserved per
micro,38 are used and26 discarded. The104 discarded indices per full update
are RNG bookkeeping, not additional training examples. Navigation has its own
Source sampler and own-h/z rollout at the same H/context/latent mode, with no
combat Source queries or future RGB inputs. NPC-motion Source data uses the
original engine/policy,64 TRAIN/16 VAL episodes with452 frames and no shooting;
continuous-life movement labels remain Source-only targets. Actor means have
explicit fixed weights: navigation Player10/64, combat Player38/64; Enemy
motion16/54 and combat38/54, excluding navigation from the NPC denominator.
Router means retain their corresponding actor shares and the existing fixed
role/balance weights. AUX/late loss weights remain unchanged.

Worker72536 launched `rssm_moe_nav15_enemy25_cap20_20261113` from full5800, SHA
`5991057b7d4df9601bc084e9c6b870936da1dd45502ffdc4a1a6041c12975f39`.
The restore check compared2267 tensors across the saved model/optimizer objects,
all277 Adam states and serialized RNG exactly. H96, stage-updates9 and MoE448
were retained. The user's further cap20 override keeps5 strict TOTAL<0.5
consecutive completed updates as the early gate, otherwise20 completed updates
per stage. Prior stage histories keep their actual old limits; final H450 stops
without implying quality. At first update5801 the actual MAIN counts were
40/64/152, AUX256/late32, and reserved draws104. No main-horizon reset occurred.

The138-test UI build accepts pinned cap20/cap25 policies separately and shows
only current revision/step counts and weighted navigation/NPC losses. Actual8765
HTTP/served-JavaScript VM/PNG proof passed27/27 at5802/H96. Atlas SR106105 and
NPR22200 remain the inference preview assets. This establishes sampling and
execution, not improved navigation/combat or sharper images. The Source-corpus
inventory and data collection did not change physics or runtime feedback.
Evidence: [launch](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/launch_command.json),
[restore](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/resume_verification.json),
[first actual update](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/diagnostics/first_three_way_update_5801.json),
[live page](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/diagnostics/live_three_way_g5802_20261113T084627Z/proof.json),
[NPC Source manifest](training_runs/rssm_enemy_motion_20261113/data/manifest.json),
[prior corpus inventory](training_runs/diagnostics/npc_motion_corpus_inventory_20261113.json).

## H74. Private renderer opacity and joint depth/RGB calibration

The user showed sharp Source walls versus rounded Base/RF at the same GT pose.
The audit found no camera-axis/FOV defect: normalized XY, sin/cos heading,
68degree FOV and forward-depth convention match Source. The renderer's frozen
CameraGeometry compresses32 columns×64 bilinear map samples into soft ray tokens.
Its learned depth was nearly constant: within-frame column standard deviation
0.051 versus Source1.506, with1.925-tile MAE on3762 exposed columns of the fixed128
panel. These measurements expose a learned conditioning failure, separate from
the RSSM pose drift. Depth, final RGB and autonomous gameplay remain distinct.

V8 adds a private copy of the original193→32→1 opacity MLP under
`flow_renderer.geometry_opacity`:6241 parameters/four tensors. Activation is
function-preserving, including Base/RF RGB and RNG. The original shared
geometry/map/RSSM is untouched. Only renderer `render_geometry` uses the copy;
dynamics' geometry path remains unchanged. Runtime still samples learned map
features; exact Source DDA intersections are detached training/validation labels,
never runtime conditions or shaders. Compact Base remains996844, RF including
the copy6998433, and frozen E7 codec997539. Full-world construction has an explicit
default-false flag; strict renderer export/load binds the private SHA to its Base
bundle and rejects missing/tampered modules or a standalone mismatched Base.

Direct first-surface NLL labels the hit and all free-prefix samples, with0.1
log-depth and0.05 clipped projected-height terms, balanced over present Source
depth bands. A500-update private-only discarded pilot reduced depth MAE
2.03208→0.28756 but RF SSIM collapsed0.77317→0.41312 on the same128 cases.
Its Base/RF were frozen and its weights were discarded. It demonstrates that
calibrated depth alone can invalidate the old learned image readout, not that
wall RGB quality was solved.

Production worker46212 instead started a fresh exact copy of completed v7/8000,
SHA `6048bfa0227eef2e4cf76d5556853602f45d3a1a3d6b307200212a2ad69e3b41`, in
`renderer_private_geometry_20261113`, target10000. Base/RF retain LR1e-4, private
LR is5e-5, and direct-depth weight ramps0→1 over500 completed updates. MAIN16,
far aux4×0.25 and geometry aux8×1 keep all three existing Source sampler clocks.
All317 old Adam states/options and RNG were preserved; only four new tensors
start with fresh Adam (321 states after SGD). No pilot weights were adopted.
The33-second real GPU preflight verified zero-activation RGB equality, all three
actual samplers, source/RNG preservation and finite gradients at first/max depth
weights. Focused CPU tests independently cover private-only and Base-RGB gradient
paths, original-world/Source-gradient isolation, strict full-world and paired
bundle roundtrips, tamper rejection and legacy behavior. The UI suite passed138.

At the first joint VAL8100, conditioning depth MAE improved2.03208→1.57261 and
projected-height MAE9.39159→6.45530px over all4096 rays/128 cases. However, main256
Base SSIM worsened0.74545→0.72376 and RF SSIM0.75799→0.73225; RF MAE increased
0.032807→0.036869. Geometry preservation and release gates remain false. The
actual PNG still shows rounded/displaced walls. No repaired-image/gameplay claim
or automatic preview/game promotion follows from the improved depth loss.
The8781 HTTP/served-JavaScript VM/PNG proof passed27/27 at8105/VAL8100/preview8100,
including private counts/LR/current ramp, both RGB/depth tables and stale-v7
rejection. Dynamics72536 and Atlas SR106105 preview continue independently.
The next VAL8200 still did not qualify: depth MAE reached1.2391 tiles, while
RF SSIM0.72661 and MAE0.038584 remained worse than the0.75799/0.032807 baseline.
The user-visible task remains in progress; smaller depth error is not sharper RGB.
Evidence: [frozen-depth audit](training_runs/diagnostics/geometry_opacity_audit_20261113/depth_audit.json),
[discarded pilot](training_runs/renderer_private_geometry_20261113/pilot/pilot_report.json),
[GPU preflight](training_runs/renderer_private_geometry_20261113/gpu_preflight.json),
[launch](training_runs/renderer_private_geometry_20261113/launch_command.json),
[VAL8100](training_runs/renderer_private_geometry_20261113/validation/step_008100.json),
[actual page proof](training_runs/renderer_private_geometry_20261113/diagnostics/live_private_geometry_g8105_20261113T085424Z/proof.json).

## H75. User-requested latest private-geometry paired game

The user explicitly requested the latest neural game. Immutable session
`training_runs/paired_latest_20261113_private_geometry` combines dynamics5805,
raw private-NPR8480 (RF cumulative24680/Base24280, private6241 parameters),
frozen E7 codec and Atlas SR106105/local5175. The NAV5152 renderer profile was
strictly rebound to combat with tensor-identical worlds, then strictly loaded
on the latest three-RSSM/MoE world through its preserved migration lineage.
All Base/RF/private/codec tensors match the export; world tensors are unchanged.

Two actual GPU smoke steps in each of navigation/combat passed. The requested
visible paired window launched as PID40072, Source left/neural right, autonomous
combat, display-size1024; its launch check recorded93/450 updates and empty
stderr. `training_runs/latest_play_session.json` selects this session. Dynamics
and NPR training continue independently; the open game uses fixed snapshots.
This is a user-requested latest-weight experiment, not a quality or15-second
gameplay pass. Evidence:
[export/rebind proof](training_runs/paired_latest_20261113_private_geometry/export_verification.json),
[profile proof](training_runs/paired_latest_20261113_private_geometry/renderer_profile_rebind.json),
[runtime smoke](training_runs/paired_latest_20261113_private_geometry/runtime_smoke_verification.json).

## H76. Latest private NPR in the dynamics preview

The user requested the latest NPR in "What the model shows". A fresh complete
raw10000 snapshot was exported as NPR26200/Base25800 with6241 private-opacity
parameters. Original NAV5152-to-combat rebinding and strict current-world5816 reload
preserved all Base/RF/private/VQ tensors. Revision `a85da445` directly follows
the Atlas revision `2535bab8`; its SR106105/local5175 entry is unchanged.

The same run resumed at5816 as PID62252. Before/after the initial H112 preview,
1434 world tensors,277 Adam states/options, all RNG/samplers and the complete
MoE gate remained bitexact, with zero SGD updates between snapshots. Live HTTP,
served-JS VM and PNG checks passed36/36. The game remains on its fixed8480
snapshot; this preview update does not establish gameplay quality. Evidence:
[export/reload](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/preview_assets/npr_private_g010000/export_verification.json),
[same-step proof](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/preview_revision_resumes/g5816_a85da4457b85/verification.json),
[live preview](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/diagnostics/live_private_npr_preview_g5816_20261113T095319Z/proof.json).

## H77. Map comparison alongside the dynamics image preview

An interactive map supplements the five image rows in "What the model shows".
It compares Source and predicted player/NPC positions, paths and directions on
the same level, with a slider for offsets0..H and the four image-offset buttons.
Distances use map tiles. NPC death, respawn and life mismatches are shown
separately; missing NPC coordinates do not become zero-error observations.

The trace is retained from the existing preview loop, including its initial
state. Source labels are read after that rollout; no extra transition, engine
query, correction or RNG draw is added. Fifteen backend/integration tests passed.
All141 dashboard Node tests and the build passed, including slider changes,
stale metadata rejection, respawn path breaks and the dead-player marker.
The actual served HTML/SVG/PNG passed32/32 live checks at5831/H128, including
all four frame buttons and slider polling persistence; [live proof](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/diagnostics/live_map_drift_g5831_20261113T111923Z/proof.json).
The same run resumed at5831/H128 as PID65836 with identical argv and image assets
(NPR26200/Base25800, Atlas SR106105). Weights,277 Adam states, all RNG/samplers and
the complete curriculum remained bitexact across the initial preview checkpoint.
See the [same-step proof](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/map_drift_preview_restart_verification.json).
The existing five-row PNG at the same step/context/H remained byte-identical;
see [image preservation](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/preview_code_restarts/g5831_a98edeb233b5/existing_preview_png_preservation.json).
An independent CPU audit matched the map and all Source coordinates to the
actual held-out episode and recomputed129 points/4 displayed offsets. In this
single H128 preview, player mean XY error was0.60689 tiles; NPC common-alive
mean was8.9008 tiles over38 points. Both trajectories respawned at offset97,
but their positions differed. These are preview diagnostics, not450-step
qualification; see the [map audit](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/preview_code_restarts/g5831_a98edeb233b5/map_metadata_verification.json).
The visualization is diagnostic and does not change the loss or gameplay rules.

## H78. Empty late Source support and durable checkpoints

The worker stopped during pending5864, micro4, at H144. Its late own-memory
branch starts at128 and queries8×16 conditional Source labels after the rollout.
The window had no legal labels, and the objective raised rather than reporting
empty coverage. Recent committed late windows were rejected for geometry;
at5863 the four micros retained80/33/77/7 of128 labels. The failed window's local
skip Counter was not persisted, so its exact actor/reason distribution is unknown.

The fix handles only three exact recoverable own-state errors:
`invalid_direct_own_state_geometry`, `noncanonical_direct_own_state_no_repair`
and `Cannot restore an entity inside a wall`; the last can include a dead NPC's
stored body position. Unknown errors still propagate. Empty windows explicitly
report attempted/valid counts, reasons and unavailable coverage. Their absent
conditional term contributes no gradient and is not a successful zero-error
observation. Main XY/router and legal AUX gradients remain active. There is no
Source correction, anchor repair, memory reset or physics/timer change. An empty
direct branch blocks the early loss streak; cap20 remains unchanged.

The durable checkpoint was5860, SHA
`2d36ac573bd5b239e283db689a66153af904928dfc650754b484dae3149e86ad`.
Status/metrics5861–5863 cannot recover their three unsaved model/Adam states;
those rows were archived rather than treated as restored weights. Pending5864
never reached optimizer.step. The runner now atomically replaces its one full
checkpoint after every completed optimizer update; preview10/full-validation250
remain separate. Core checks passed46, including8 coverage/gate regressions;
the broader84-check baseline and143 Node tests/build also passed. The dashboard
marks empty counts/unavailable support beside MoE losses and rejects stale metrics.

A bounded read-only GPU scout from5860 took12.67s: the first64 episode draws,
38 combat actors, H144/context2/argmax and128 late queries produced74 valid and54
geometry-invalid labels, matching every recorded5861/micro1 support count.
Weights, checkpoint and CPU/CUDA RNG were unchanged; no backward, Adam, AUX
rollout or Source feedback was used. This is not an exact replay of failed5864.
Recovery resumed from5860 as PID67840. Before/after its initial preview save,
all1434 world tensors,277 Adam states/options, all RNG/three sampler PCGs, gates,
argv and revision`a85` matched exactly with zero SGD between snapshots. The first
recovered5861 update then completed and was atomically saved: all1434 tensors and
277 Adam states were finite, frozen assets remained bitexact, H144 count reached10,
and all64×4 quotas/sampling receipts matched. Loss1.7285662423635346, Player/Enemy/
Combat/router losses and gradient norm exactly matched the archived5861 row.
Training proceeded to5862. This confirms the first durable update, not gameplay
quality; [first commit proof](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/failure_recovery_resumes/g5860_failed5864_45182ee251cc/first_recovered_commit_verification.json).
Evidence:
[boundary/failure audit](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/failure_recoveries/failed5864_20261113T140017Z/failed_update_consistency.json),
[GPU scout](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/diagnostics/direct_label_failure_20261113/scout5860.json),
[same-step recovery](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/failure_recovery_resumes/g5860_failed5864_45182ee251cc/verification.json).

## H79. Microbatch16 with four-step accumulation

The user reduced the dynamics microbatch to16, keeping accumulation4 and
effective MAIN64. Protocol `rssm_moe_micro16_accum4_v1` pins full source5863,
H144/stage12/MoE511, with `micro16_batch_revision_plan.json` (`e176b5…`).
All existing model/277 Adam states, RNG/private PCGs and gate/streak/history are
retained. Historical256-example counts keep their actual size; new updates add64
using a separate revision ledger. No architecture, memory or task-weight reset occurs.

Combat/NPC-motion/navigation micros are9/4/3,10/4/2,9/4/3,10/4/2: effective38/16/10,
retaining59.375%/25%/15.625%. Player/router use actual actor shares; enemy weights
use the global54-actor NPC denominator, excluding NAV. Each micro now draws16
combat indices and retains9/10; discarded reservations7/6 total26 per update and
are not learning examples. Initial PCG states are preserved; the new draw sizes
intentionally change subsequent sampling. AUX16/micro gives64 per update and
late2/micro gives8, both outside MAIN shares.

AUX16 retains quarters4: rotating least-used spawn anchors, positive attacks
2 near-wall/2 open, negatives3 occluded/1 other, shooting near/mid/far1 each plus
one alternating genuine miss/legacy. Four draws yield4/4/4/2/2 target counts;
wall/other misses rotate on actual miss draws. Old anchor/positive/legacy counters
remain intact. CPU tests verify new16 behavior and exact old32/64 pairs, receipts,
ledgers, summaries and PCG. Original Source-pool/sampler identities stay pinned;
the new batch plan separately declares runtime16 sampling. Backend checks passed60;
147 Node tests and the build passed.

The22-check disposable GPU preflight passed in150.07s at H144 across all four
micros: MAIN64=38/16/10, AUX64 and late8, with finite positive gradients through
all three RSSMs/MoE. It performed zero SGD and preserved weights/277 Adam states.
Production resumed from5863 as PID50676. Initial before/after proof matched all
1434 world tensors,277 Adam states, configuration, all RNG/three PCGs and the
H144/stage12 history/streak exactly. Only batch/navigation/runtime metadata changed;
revision updates remained0. The first16×4 update5864 then committed and was saved:
MAIN64=38/16/10, AUX64 and late8, all1434 world tensors/277 Adam states finite,
frozen assets bitexact and piecewise old256/new64 counts correct. Loss was
1.2956592584, gradient norm38.05797, H144/stage13. Two actual empty direct branches
were reported without aborting; their missing support correctly blocked the
early loss gate. Stderr remained empty and5865 started; see
[first durable commit](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/batch_revision_resumes/g5863_e176b5d629df_9fbca9fd/first_micro16_commit_verification.json).
Checkpoint-every-update, cap20/early five strict TOTAL<0.5 passes, empty-support
gate protection and frozen NPR26200/Atlas106105 remain unchanged. No gameplay
quality or VRAM-improvement claim follows from activation. Evidence:
[batch plan](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/micro16_batch_revision_plan.json),
[GPU preflight](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/micro16_gpu_preflight.json),
[initial activation](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/batch_revision_resumes/g5863_e176b5d629df_9fbca9fd/verification.json).

## H80. Latest neural game after the micro16 continuation

The requested visible paired game launched from immutable inference snapshots in
`training_runs/paired_latest_20261113T155600Z`: dynamics5896, private NPR raw10000
(RF26200/Base25800) and Atlas SR106105. Strict loading and two actual GPU steps
in each of navigation/combat passed with5232-value RSSM memory and private opacity.
PID60812 opened a1572×968 window with768-pixel fitted panels; the neural display
pipeline still produces1024×1024 before panel fitting. Source is left, neural
right; T selects combat and E turns right. Readiness was confirmed by the window
handle and live status; stderr was empty. Dynamics training continued independently.
See [snapshot](training_runs/paired_latest_20261113T155600Z/snapshot_receipt.json)
and [runtime smoke](training_runs/paired_latest_20261113T155600Z/runtime_smoke_verification.json).

## H81. Separating image quality from dynamics in the played5896 game

Read-only evaluation used the exact5896/RF26200/Base25800/SR106105 game assets,
two held-out450-step tapes on the same default map, and render-only Source-pose
interventions with unchanged own h/z features. Within the trained H176 window,
navigation player mean XY error was0.14866 tiles; six image samples had native
SSIM0.39892, versus0.39716 with exact Source poses and0.86390 for the empirical
codec reconstruction reference. Walls already differed at zero-drift frame0.
Combat separately had NPC mean error8.73953 tiles over85 supported points and
an11-tile error at the first spawn. Renderer/source conditioning is therefore
a major visual bottleneck; enemy dynamics remain independently wrong.
The same256-pose renderer panel also regressed from SSIM0.75799 at raw8000 to
0.68820 at raw10000. Suggested follow-up: current-RSSM conditioning, fixed
geometry/codec while Base/RF adapt, patch1 ablation, then refreshed SR pairs.
No training or gameplay settings changed. [Assessment](training_runs/diagnostics/render_vs_dynamics_g5896_20261113T160740Z/summary.json).

## H82. Renderer token MoE with a private Transformer per expert

The user requested renderer improvement in parallel with dynamics, then explicitly
asked for a separate Transformer block for every expert. The opt-in implementation
adds three soft per-screen-token roles, scene/walls, enemy and combat, to both
compact Base and RF. Each expert owns an independent block after a shared small
input projection: Base width8/heads2/FF16, RF width32/heads4/FF64, dropout0.
The shared output is zero-initialized; existing shared Transformers and all old
renderer weights retain their function at activation. Base has999,939 parameters,
RF7,034,500 including private6241 geometry;39,162 parameters in84 tensors are new.

Worker76384 runs `training_runs/renderer_moe_20261113` for2000 local updates from
the complete private10000 parent, cumulative RF26200/Base25800. All321 old Adam
states/groups, RNG and three Source sampler states are preserved; the new expert
group starts fresh. MAIN16, far4×0.25 and geometry8×1 remain. The distinct frozen
feature snapshot is current combat-world5906: previous-RGB/state/action priors use
the recorded navigation/combat profile; static features are zero. World/map, E7,
SR and the private opacity MLP stay frozen. Source role masks supervise balanced
router CE×0.01 and balancing×0.001 only; inference receives no future Source data.

39 CPU tests passed. The actual7.17-second GPU probe on5906/private10000 verified
exact initial Base/RF RGB, exact strict export/reload, all old tensors/RNG unchanged
and finite positive router/readout gradients, with0 optimizer updates and0.622GiB
peak allocation. Export retains the pristine world as the original codec source
and validates E7 through its compact-codec provenance; no lineage bypass is used.
An independent audit of the durable local34 checkpoint confirmed405 finite Adam
states, unchanged frozen assets/private moments and exact sampler draw clocks.
All84 new tensors, including all six private Transformer blocks, received gradients;
this audit observed34, not the overwritten initial0/1 checkpoints.
The actual training process reached113 updates/preview110 without errors. Native32
preview cadence is10, held-out validation100. On the same256-frame panel with the
same5906 feature policy, initial→local100 RF SSIM improved0.652124→0.676553 and MAE
0.057161→0.049310; Base SSIM improved0.63298→0.65812. Quality and geometry-preservation
gates remain false. The old10000 score used5152 features and is not a matched
comparison for this phase. **http://127.0.0.1:8781/** now monitors
this phase. Dynamics training continues independently. The game/dynamics preview
keep their existing NPR26200 and Atlas106105 pins until an explicit asset revision;
no quality pass or automatic promotion is claimed.

Evidence: [launch](training_runs/renderer_moe_20261113/launch_command.json),
[frozen inputs](training_runs/renderer_moe_20261113/inputs/snapshot_receipt.json),
[GPU warm-start](training_runs/renderer_moe_20261113/gpu_warmstart.json),
[matched initial validation](training_runs/renderer_moe_20261113/validation/step_000000.json),
[VAL100](training_runs/renderer_moe_20261113/validation/step_000100.json),
[durable34 audit](training_runs/renderer_moe_20261113/diagnostics/independent_first_commits/verification.json).

## H83. Five renderer experts with preserved parent routing

The user split renderer work into near geometry/player, far geometry, enemy
motion, enemy combat and player combat, with a private Transformer for each role
in both Base and RF. The preceding three-role run completed2000 updates, cumulative
RF28200/Base27800. Its final256-frame RF SSIM/MAE/foreground MAE was
0.701969/0.043726/0.097023; source quality and geometry guards remained false.
Relative experimental selection was true, which is a separate criterion.

The opt-in five-role schema retains the three learned parent gates and all
existing shared weights. Two new sigmoid splits partition scene into near/far
and enemy into motion/combat; cloned private blocks preserve the initial parent
function. Canonical order is `near_geometry_player`, `far_geometry`,
`enemy_motion`, `enemy_combat`, `player_combat`. Each expert has one block:
Base width8/heads2/FF16, RF width32/heads4/FF64. Totals are1,001,157 Base and
7,051,654 RF parameters; 52 new tensors add18,372 parameters. All405 old Adam
states/options, serialized RNG and three sampler streams are retained; only
the appended group starts without moments. Inference/export uses the distinct
version2 bundle schema; the old three-role format remains supported.

Source role labels use near<5 tiles/far≥5. Idle gun pixels label near/player;
flashes, shots and hurt effects label player combat. The full frame is a combat
label only for player HP0. These are detached training/validation labels, with
no runtime Source DDA. Per-token hierarchical soft routing uses existing neural
tokens/current numeric state. CE×0.01 and balance×0.001 remain. MAIN16,
far4×0.25 and geometry8×1, feature world5906, frozen RSSMs/map, E7/SR and private
opacity6241 are unchanged.

Worker38908 started `training_runs/renderer_moe_five_20261113` for2000 local
updates in parallel with dynamics. 25 CPU tests passed. The actual7.72-second
zero-SGD GPU preflight verified exact initial Base/RF latent/RGB and strict
export/reload, unchanged old tensors/RNG, and positive gradients in all ten
private expert blocks. The durable local47 audit verified401 active inherited
Adam clocks advanced47, four private-opacity Adam states bitexact, all52 new
moments positive, finite weights and exact sampler increments752/188/376.
Initial step0/1 were not independently captured; restoration is attested by
the immutable phase plan and GPU preflight. The live HTTP/VM check passed at
82/preview80, showing ten effective role means plus actual parent/split means.
It does not multiply means to invent effective gates. Previews run every10,
full validation every100 on256 fixed held-out Source poses plus the128-frame
geometry guard. Initial source quality is not passed; improvement from the new
phase and autonomous gameplay quality are not claimed. The8781 pointer now
follows this phase. Pinned game/dynamics-preview assets remain unchanged.

Evidence: [launch](training_runs/renderer_moe_five_20261113/launch_command.json),
[phase plan](training_runs/renderer_moe_five_20261113/renderer_moe_five_phase_plan.json),
[GPU preflight](training_runs/renderer_moe_five_20261113/gpu_warmstart.json),
[baseline](training_runs/renderer_moe_five_20261113/validation_initial.json),
[durable47](training_runs/renderer_moe_five_20261113/diagnostics/independent_first_commits/verification.json),
[live UI](training_runs/renderer_moe_five_20261113/diagnostics/independent_first_commits/live_ui_proof.json).

## H84. Five private dynamics experts over three preserved RSSMs

The user also requested five specialized dynamics experts. A new opt-in adapter
follows the pretrained shared three-role trunk; the three recurrent player,
enemy and combat RSSMs remain unchanged. Canonical roles are
`player_without_enemy`, `player_with_enemy_quiet`, `player_with_enemy_combat`,
`enemy_motion_spawn`, `combat_health_hit_death`. Five private Transformer blocks
use width256/heads4/FF512 and dense soft routing. The first three role heads
refine player pass-axis logits; the enemy head refines NPC motion/spawn and the
combat head refines shot-target/attack logits. HP, hits and deaths still follow
the existing structured transition. Zero new task readouts preserve the initial
transition; there is no extra recurrent memory or future Source input.

The adapter adds2,717,468 parameters in83 tensors. A12.52-second actual GPU
preflight on the frozen5937 snapshot preserved16 navigation and16 combat
transitions, full5232 memory, states/events/logits/features and the strict
NPR26200 conditioning lineage. Actual production adoption used the later complete
5939 boundary, not that earlier checkpoint. Its zero-SGD verification preserved
all1434 old tensors,277 Adam states/four groups and their mixed ages, RNG, the
complete RSSM training subtree and all sampler/curriculum/asset metadata. Only
one appended optimizer group starts without moments; all previous histories and
the original CLI remain intact.

Worker73164 resumed the same run at5939/H208 with cap20, micro16×4 and the
38combat/16NPC-motion/10navigation main allocation. Pinned NPR26200/Base25800
and Atlas SR106105 remain unchanged; the parallel five-role renderer is not
automatically promoted. The initial8765 HTTP/VM check passed at source/current
5939: five expert rows are visible, routing is unavailable until the first
matching completed update, and there are still three RSSMs and three task-loss
groups. New routing diagnostics average actual own forward calls/queries; they
are not the weighted optimizer objective or evidence of gameplay quality. Old
and new router auxiliaries use the existing0.01/0.001 weights.

The first complete update5939→5940 passed15 independent checks: all277 old Adam
states retained,269 active ages advanced once, eight dormant states bitexact;
all83 new states are finite at local age1, and1157 frozen tensors are unchanged.
It consumed four16-example micros, MAIN38combat/16motion/10navigation plus
AUX64/late8. H208 stage updates advanced8→9 and the MoE clock587→588 without reset.
TOTAL was3.012795. The measured five-expert importance was0.2 each, with player
mode means0.204365/0.365968/0.429670 over2624 own forwards,72,320 role queries
and14,464 actor-ticks. These means are not the training-mixture percentages.
New router/player/enemy/combat readout gradients were positive; all five private
block gradients were zero on this first zero-readout step, as expected.
Although61 new tensors changed and14 had nonzero moments, this is not evidence
that the private blocks have already learned. First-update HTTP/VM checks passed;
no navigation, combat or image-quality improvement is claimed.

Evidence: [bounded GPU preflight](training_runs/diagnostics/rssm_five_preflight_20261113/verification.json),
[migration/start archive](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/five_role_revisions/02ec0c0c2513_20261113T175138474127Z/start.json),
[zero-SGD preservation](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/five_role_revisions/02ec0c0c2513_20261113T175138474127Z/verification.json),
[initial UI](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/five_role_revisions/02ec0c0c2513_20261113T175138474127Z/live_initial_ui_proof.json),
[first complete update](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/five_role_revisions/02ec0c0c2513_20261113T175138474127Z/first_update.json),
[committed routing UI](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/five_role_revisions/02ec0c0c2513_20261113T175138474127Z/live_first_update_ui_proof.json).

## H85. Mandatory100 updates at H240 H368 and H450

The user requested100 complete optimizer updates specifically at H240, H368 and
H450. Policy v5 disables early loss advancement only on those three stages.
Every other horizon keeps cap20 OR five consecutive strict TOTAL<0.5 decisions.
Final H450 ends after100 with `completed_budget`; budget completion is not quality.
The stages are future work, not100 updates already executed. Historical policies
and completed stage counts remain intact.

Worker73772 resumed the same5947 boundary with H208/stage16, MoE595 and streak0
preserved. The zero-SGD27-check audit retained all1517 model tensors,360 Adam
states/five groups, RNG and all three sampler PCGs; the complete training state
matches the explicitly updated policy and preview revision. Model weights were
not reset. The policy plan is `stage_budget_horizon100_plan.json`, SHA
`bb116e8c04df6b303bcf7ec999394ef7281e2ef155d22eba611d3cea28439cdd`.
Live8765 HTTP/VM checks confirm current16/20, future mandatory100 flags, unchanged
16×4 and DYN five-role pins. UI simulations are labelled as future semantics,
not measured training. 161 Node tests and the dashboard build passed.

Evidence: [same-step preservation](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/budget_horizon100_revisions/g5947_1dbef55a2120/initial_resume_verification.json),
[live combined UI](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/budget_horizon100_revisions/g5947_1dbef55a2120/live_combined_ui_proof.json).

## H86. Independent validation of five-role renderer1156

A read-only snapshot at local1156 (RF29356/Base28956, frozen feature world5906)
was evaluated on256 held-out Source poses,128 geometry cases and192 distance
cases, with zero TRAIN-scene overlap. Validation took51.68 seconds, performed
zero optimizer updates and preserved all weights/input files and live training.
Main-panel RF SSIM improved0.701969→0.730221, MAE0.043726→0.037521 and NPC foreground
MAE0.097023→0.092395 (142 visible cases). Base reached SSIM0.715937/MAE0.038201.
The frozen codec empirical reference was0.888866/0.021369, not a proven ceiling.

Absolute quality, Base preservation, geometry and distance guards remained false;
relative experimental selection was true. On64 far-NPC cases, RF foreground MAE
worsened0.081721→0.083856. Aggregate improvement therefore does not establish
improved distant enemies or sharp wall geometry. SR and autonomous gameplay
were not evaluated. [Pinned report](training_runs/renderer_moe_five_20261113/diagnostics/validation_20261113T181806Z/summary.json).

## H87. Requested five-role NPR in the dynamics preview

The user stopped renderer training and explicitly requested its latest weights
for dynamics visualization. Renderer worker38908 exited at local1409; export
RF29609/Base29209 retains five private experts and frozen private opacity6241.
It remains experimentally unqualified. Its immutable preview bundle SHA is
`fcd8a86e5e9574d5d1939c33859ed3b8747aef7718c6393c1a070865d122c80f`.
Atlas SR106105 and its exact SHA6e90… are unchanged. The visible game is unchanged.

The same5947 restart changed exactly two CLI values: the selected-horizon budget
plan and the preview-assets revision plan. File suffix`_v3.json` denotes the
third selection, while the validated schema remains `preview_assets_revision_v2`,
chained to the earlier a85da445… revision. New revision SHA is
`8d4b3ab2b67bc5738139aef4461e95a836e505f5f236dc1567cfeebec667e030`.
The explicit user selection changes the separate frozen rendering clone, not
dynamics weights/Adam or training inputs. Initial preview5947 generated all208
own transitions and retained five rows plus the same read-only drift-map trace.
HTTP/VM22/22 checks passed: RF29609/Base29209/Atlas106105, five-role metadata,
matching pins and zero Source-future feedback. No quality promotion is claimed.

Evidence: [export verification](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/preview_assets/npr_moe_five_g001409/export_verification.json),
[revision plan](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/preview_assets_revision_npr_moe_five_29609_atlas_sr106105_v3.json),
[live preview proof](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/budget_horizon100_revisions/g5947_1dbef55a2120/live_combined_ui_proof.json).

## H88. Five private SR experts on refreshed NPR pairs

The user requested five SR experts for enemy recognition, sprite choice and
placement, far walls, near walls and player/weapon effects. Each has a private
Transformer (width64,4 heads,FF128,grid8). The new phase
`sr_moe_20261113/training` preserves the frozen AtlasSR106105 parent and all160
old Adam states; only81 new tensors/343,460 parameters are trained, bringing
the total to2,311,333. Native RGB/visibility heads operate at128 and the heat
head at32. Dynamics and the stopped renderer are not trained by this worker.

The original8192 TRAIN/1024 VAL Source128 pairs remain unchanged. NPR32 and
one-step causal hypotheses were rebuilt from world5947 and stopped five-role
RF29609/Base29209, with Source environment provenance resolved for all9216 rows:
553 dynamic NAV,6183 dynamic combat and2480 static combat. Static hypotheses
are zero/invalid. Future Source images, HR and masks are labels only; they do not
condition NPR features or predicted hypotheses. The new private sampler starts
at seed173/draw0; the parent's sampler is retained as history, not presented as
the same new sampling clock.

The actual GPU warmstart passed, including three discarded SGD steps and
positive gradients for all five new experts. Production worker43404 started.
Live HTTP/served-JavaScript VM/PNG checks passed21/21 at local1005, fullVAL1000
and preview1005: exact asset/data pins, frozen parent, five measured gate values,
four preview rows and actual current train/weighted NPR validation loss.
The existing8779 server was left running. It lacks the new full VAL-history
loader, so the client adds only the latest strictly verified report loss and
discloses that limitation; MAE/SSIM and reference scores are never used as loss.
Light preview's reference0 is displayed separately from fullVAL1000.

The strict goal remains actual NPR SSIM>0.91 and MAE255<2.5; it has not passed.
Routing means and a completed budget do not establish specialization or gameplay
quality. The current game and dynamics retain AtlasSR106105; no automatic
promotion occurred. Evidence:
[environment inventory](training_runs/sr_moe_20261113/diagnostics/source_environment_inventory.json),
[GPU warmstart](training_runs/sr_moe_20261113/gpu_warmstart.json),
[live UI proof](training_runs/sr_moe_20261113/diagnostics/live_ui/verification.json).

## H89. Microbatch32 with one accumulation step

The user changed dynamics to32×1=effective32. The one MAIN micro contains
19 combat,8 pure-NPC-motion and5 no-enemy-navigation actors, preserving
59.375%/25%/15.625%. Combat AUX32 and late own-prefix4 are separate from MAIN.
The event-balanced AUX sampling policy is explicitly revised for32; old clocks
and counters remain part of the continuation rather than being reset.

Worker75404 resumed the same run at5958/H224/stage7. Only the saved
`--batch-revision-plan` value changed. New plan SHA
`fa8017e91e294e71cd56c7c6f815932805980917864b8f796ed11ec4240960a6`
pins the exact micro16 parent plan. Same-step verification passed22/22 with zero
SGD:1517 model tensors,360 Adam states/five groups, all RNG and three sampler
streams, expected gate/protocol/config and immutable inputs were preserved.
The previous256→64 ledger remains nested and is not rescaled to32.

Actual update5959 completed one flat32-example micro, advancing H224/stage7→8.
HTTP/JavaScript VM checks passed15/15 both before SGD and after that commit:
MAIN19/8/5, separate AUX32/late4, batch32×1 and recursive historical counters.
The UI accepts the one-micro metric record without inventing a four-row
accumulation group. Before the first commit it showed the planned mix while
withholding old64-example measurements from the new actual-count fields.

Mandatory100 updates at H240/H368/H450, other-stage20 OR five strict TOTAL<0.5
passes, the three RSSMs/five experts and frozen RF29609/Base29209/AtlasSR106105
remain unchanged. No quality improvement is inferred from changing batch size.
Evidence:
[same-step proof](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/micro32_revisions/g5958_c7331a75b66d/initial_resume_verification.json),
[first-update UI proof](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/micro32_revisions/g5958_c7331a75b66d/live_first_update_ui_proof.json).

## H90. Completed SR validation, preview deployment and dynamics recovery

SR5 completed2000 local updates/cumulative108105. Independent fixed256 validation
used the refreshed NPR inputs and cached one-step causal hypotheses, not an
autonomous450-step game. Against its frozen Atlas106105 parent, SSIM improved
0.761640→0.762556 and MAE2559.726→9.614. Enemy foreground MAE worsened
0.158270→0.160411 (205 frames), foreground edge0.182256→0.186701,
far-wall edge0.052267→0.054808 (188 frames), and false NPC detections changed
0/51→2/51 empty frames. The image goal SSIM>0.91 AND MAE255<2.5 remains false.
Validation integrity passed with zero SGD and unchanged SR/reference/sampler;
this is a mixed experimental result, not a quality promotion. See the
[independent report](training_runs/sr_moe_20261113/diagnostics/validation_20261113T192236Z/summary.json).

The user explicitly requested this SR in dynamics' “What the model shows”.
Self-contained export SHA3127bef2… retains embedded Atlas pixels and the five
private Transformer experts. Preview revision044e0811… changes only the frozen
SR selection to108105/local2000; NPR29609/Base29209 remains exact. Dynamics
previews call SR with `none_RGB_only`, without cached hypotheses. The validation
numbers above therefore describe a different input-hypothesis scenario and are
not presented as measurements of the displayed autonomous frames.

A separate fixed256 RGB-only evaluation then matched the dynamics-preview SR
call: SSIM0.763592→0.764103 and MAE2559.638→9.531 versus the frozen parent.
Foreground MAE0.149978→0.153620 and edge0.168969→0.175860 worsened; near-wall
MAE0.055005→0.055175 and far-wall0.056358→0.057505 also worsened. Both models had
0/51 empty false-NPC frames. The strict image goal remained false; this measures
cached NPR/SR quality, not autonomous dynamics or gameplay. See the
[RGB-only report](training_runs/sr_moe_20261113/diagnostics/rgb_only_validation_20261113T193747Z/summary.json).

Meanwhile the dynamics process was interrupted around5972 with a damaged rolling
checkpoint (NUL tail) and an all-NUL status file; the cause is unknown. An older
Source exception predates this worker and does not explain this interruption.
The damaged files and orphan5972 metrics were archived under
`failure_recoveries/20261113T193007.035316Z_g5971`. Recovery used the exact saved
5971 snapshot (SHAf01217e8…) and preserved1517 model tensors,360 Adam states,
all RNG/three sampler streams, batch32×1 and the v5 mandatory100 policy.
Independent same-step verification passed28/28. The replayed5972 update passed
27/27, with finite loss2.959626062313931,1157 frozen tensors exact and all
nonoperational metrics/Source-pair receipts identical to archived5972.
No game-engine modification is claimed by this recovery.

Live8765 HTTP/JavaScript VM/PNG checks passed17/17 at current5976/preview5971H240:
SR108105/local2000/five private Transformers, unchanged NPR/Base, five image rows,
the same map trace, zero future feedback, batch32×1 and mandatory100 intact.
The absent8765 listener was separately restored; the existing8779 SR server was
not restarted. Evidence:
[resume audit](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/preview_revision_resumes/g5971_044e081134af/independent_verification.json),
[replayed update](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/preview_revision_resumes/g5971_044e081134af/first_update.json),
[live preview proof](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/preview_revision_resumes/g5971_044e081134af/live_ui_verification.json).

On the subsequent explicit launch request, a fresh paired window opened as
`paired_latest_20261113T193445Z` with immutable dynamics5974/NPR29609/Base29209/
SR108105 snapshots: Source left, neural game right, T toggles combat and E turns
right. It initially paused at0/450; the visible game is a separate snapshot and
does not receive later training weights automatically. Gameplay remains
unqualified. See its
[runtime smoke](training_runs/paired_latest_20261113T193445Z/runtime_smoke_verification.json).

## H91. Explicit five-Euler-step RF playback

The user requested RF playback with five integration steps. The new runtime
option `--rf-euler-steps 5` is applied after strict checkpoint loading; the
saved model and validated training configuration remain RF1. Actual velocity
evaluations use Euler times0/0.2/0.4/0.6/0.8. Four focused ODE tests and seven
RSSM/control regressions passed. The runtime smoke recorded five velocity
evaluations on each of four transitions: `[5,5,5,5]`.

New paired session `paired_latest_20261113T200421Z` (PID44096) reuses the exact
previous immutable dynamics5974/NPR29609/Base29209/SR108105 assets. This changes
inference integration only: no five-step fine-tuning or expert widening was
performed. More RF evaluations cost additional rendering time; quality remains
unqualified. See its
[runtime smoke](training_runs/paired_latest_20261113T200421Z/runtime_smoke_verification.json).

The separate capacity discussion was advice, not an architecture-change request.
Each current private Transformer has600 parameters in Base and8544 in RF.
Replacing each of five blocks by approximately0.5M parameters would bring Base
to roughly3.50M and RF to9.51M, or13.01M combined before further adapter changes.
The current implementation evaluates all five experts; wider blocks would
increase runtime work. These sizes have not been implemented or validated.

## H92. Expanded RF experts trained with three Euler steps

The user authorized approximately0.5M parameters per RF expert, then selected
three Euler steps instead of five. Each of the five experts now retains its
8544-parameter legacy block and adds a private width256/heads8/FF416 Transformer
with zero-initialized residual output:503,072 parameters per expert.
RF totals9,524,294 excluding codec. Base, VQGAN, private opacity and the frozen
feature world5906 remain unchanged; no Base expansion or GAN training is used.

The new run `renderer_rf_expanded_20261113` continues the full stopped1409 parent
RF29609/Base29209:457 Adam states/five groups are retained,80 tensors enter a
fresh sixth group, and all three Source sampler/RNG clocks continue. Budget2000,
MAIN16/far4/geometry8, preview10 and VAL100 keep the original held-out panels.
The new nested format2 bundle validates its RF1 parent and declares actual RF3;
legacy RF1/RF4 validation remains strict. It does not rewrite the parent config.

The first zero-SGD export exposed a real lineage error: revalidating an already
compact-codec working model against original-teacher lineage. Export now receives
the untouched original `source_model`, validates the parent there, and separately
checks working conditioning/frozen assets. Restart occurred at0; no SGD was lost.
Independent GPU checks passed19/19 with0 SGD: old/expanded RF1 and RF3 initial
RGB/latents were bitexact, and the three-discarded-update RF3 bundle reloaded
bitexactly. Inputs, Base/codec/private tensors and global RNG were unchanged.

Production PID40968 was verified at local18/preview10/VAL0. Live HTTP/VM/PNG
checks passed17/17, including three actual velocity calls and gradients in all
five experts. Main loss0.268245 and total0.381563 are separate; VAL0 main loss
was0.315476. Its SSIM0.729578/foreground MAE0.104547 versus parent RF1
0.735882/0.094481 is a solver comparison, not an SGD improvement. Quality remains
false. [Live page](http://127.0.0.1:8781/),
[GPU bundle proof](training_runs/renderer_rf_expanded_20261113/gpu_bundle_verification.json),
[live UI proof](training_runs/renderer_rf_expanded_20261113/diagnostics/live_ui/verification.json).

The separate open three-Euler game `paired_latest_20261113T202309Z` retains
the old dynamics5974/NPR29609/Base29209/SR108105 snapshots. Newly trained expanded
RF weights have not been promoted into the game or dynamics previews.

## H93. One million parameters per private dynamics expert

The user separately requested approximately1M parameters per dynamics expert,
not the RF renderer's0.5M profile. Each of five dynamics experts now has
1,005,088 parameters: its527,104-parameter Transformer is retained and a second
private FF288 block/zero-output branch adds477,984. Total five-role fusion is
5,107,388;2,389,920 parameters/70 tensors are new. The three RSSMs and5232-value
h/z memory, original task losses and shared trunk remain intact.

Worker45296 continued source6022/H240/stage51 of100, preserving old360 Adam
states/five groups, all Source samplers/RNG, batch32×1 and the v5 stage policy;
the new70 parameter tensors enter a sixth group. Initial verification passed10/10:
durable migration and the actual trainer restore were measured BEFORE startup
preview. The separate after-preview6022 checkpoint capture window was missed;
that file is not claimed as evidence. The first captured worker checkpoint is6023.

Actual6023 passed24/24:1587 model tensors/430 Adam states finite,352 old moment
ages advanced once,8 dormant states remained exact,70 fresh states reached
local1 and1157 frozen tensors were bitexact. Loss2.994952 and gradient norm90.71459
were finite; H240 advanced51→52 with mandatory100 retained. All five new output
gradients were positive; the new deeper blocks had zero gradients on their
first update while zero-initialized output projections opened. Their second-step
gradient path had already passed the CPU test. Committed6024 and6025 then logged
positive finite gradients for all five new private Transformers; this confirms
that their training opened, not that image/gameplay quality passed.

Whole preview revision044e…/selected plan18e425… and exact NPR29609/Base29209/
SR108105 file pins stayed unchanged. Four measured CACHE provenance fields
updated for the new architecture: current world6020→6022, copied tensors551→621,
initialization5971→6022 and source-model SHA. These are not asset revisions.
Live8765 HTTP/VM/PNG checks passed16/16 at6023/preview6022, including exact1M
metadata,32×1, five image rows/map, mandatory100 and zero future Source feedback.
The open game and preview do not receive the separately trained expanded RF3.
An independent8.80s/zero-SGD compatibility probe combined dynamics6023 with
expanded RF3 local100/cumulative29709 and SR108105: four own combat transitions,
5232 memory and32→128→1024 images were finite, without future-engine feedback
or asset promotion. This checks runtime compatibility, not quality; see
[combined probe](training_runs/diagnostics/expanded_dynamics_rf3_sr_compat_20261113T204613Z/verification.json).
Evidence:
[initial restore](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/expanded_expert_revisions/g6022_874260e9df60/initial_verification.json),
[first update](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/expanded_expert_revisions/g6022_874260e9df60/first_update.json),
[live UI](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/expanded_expert_revisions/g6022_874260e9df60/live_first_ui_verification.json),
[subsequent gradients](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/expanded_expert_revisions/g6022_874260e9df60/private_transformer_gradients.json).

## H94. Independent validation of expanded RF3 at local854

On14 November, immutable snapshot854/RF30463/Base29209 was independently
evaluated on the reused fixed Source panels: main256, geometry128, native-area
guard256 and distance192. These are native32 GT-pose queries with frozen5906
causal past-prior/static-zero features, not current autonomous dynamics, SR or
450-step gameplay. Integrity passed:0 SGD, zero TRAIN-scene overlap and unchanged
model/input files, samplers and global RNG.

On main256, RF3 initial→854 SSIM improved0.729578→0.741267,
MAE0.037757→0.036027 and NPC foreground MAE0.104547→0.092844 (142 visible).
Original RF1 reference was0.735882/0.036579/0.094481; it is a separate solver
reference. Absolute image quality and relative selection both remain false.
Six of60 geometry checks regressed versus RF3 initialization: near<2 corner,
8–12 cross-boundary contrast, and12+ wall/face/height/contrast. Other aggregate
improvements do not cancel these failures. The optional18-frame HP0 supplement
was omitted because its separate dataset indexing failed the strict fingerprint.
No model or game assets were promoted. Evidence:
[summary](training_runs/renderer_rf_expanded_20261113/diagnostics/validation_20261113T214803Z/summary.json),
[wall cases](training_runs/renderer_rf_expanded_20261113/diagnostics/validation_20261113T214803Z/worst_wall_boundary_cases.png).

## H95. Decoder-only adaptation on paired RF and Source codes

The user authorized training only decoder/post_quant_conv:528,403 parameters,
62 tensors and fresh Adam. Frozen RF3 stopped at1108/cumulative30717; Base29209,
original world5906, encoder/quantizer/codebook, normalization and noise stay exact.
Each8 Source queries yields8 RFQ plus8 SourceQ examples (16 decoder examples,
50/50 loss). The old decoder remains a separate frozen reference. New format3
nests the strict RF3 parent and overrides only complete D/postquant weights;
RF counters remain fixed and decoder updates are separate. The GPU export/reload
probe passed25/25 with0 SGD after three discarded warm updates: all62 D tensors
changed, both RGB branches changed, RF z/EQ codes and every non-D tensor stayed
exact; fresh format3 loading was bitexact. No codec qualification is inherited.

Worker49828 runs `decoder_rf_adaptation_20261114` on8782, budget2000,
preview10/VAL100. Initial four-branch evaluation and geometry128 are separate.
On fixed256 VAL100, RF old→new SSIM0.741410→0.741162, MAE0.036166→0.036655,
foreground MAE0.095424→0.097506; Source old→new SSIM0.888866→0.887124,
MAE0.021369→0.021874, foreground0.079946→0.081209. Source-preservation failed
geometry and RF-improvement failed foreground/geometry; quality remains false.
Training continues, while dynamics and existing game/preview image pins remain
unchanged. This native32 GT-pose validation measures neither SR nor autonomous
gameplay. Evidence: [GPU proof](training_runs/decoder_rf_adaptation_20261114/gpu_bundle_verification.json),
[VAL100](training_runs/decoder_rf_adaptation_20261114/validation/step_000100.json).

## H96. Conditional restorer over a frozen RF3 and decoder

The user authorized a separate learned restorer with extra current-state/map
hints. It has247,539 parameters/65 tensors, two width64 attention blocks,
4 heads and FF128. Inputs are RF3 quantized16×8×8 codes, decoded RGB32,
current state20 and known raw16×16 material IDs; runtime uses no Source image,
mask, raycast or engine oracle. Zero last convolution preserves the initial RGB.
Only this module trains; RF30717/local1108, Base29209, world5906, E/Q/codebook,
decoder/postquant1725 and all parent assets are frozen. Training uses8 aligned
Source queries→8 RF-code+8 Source-code examples,50/50, with fresh65-state Adam.

Strict format4 nests the unchanged format3 decoder parent, binding full module
config/state/hash and a separate restorer counter. The runtime sample hook runs
after actual RF3→denormalization→Q→decode with the same current state/raw map;
plain Source-code codec.decode does not apply the restorer automatically.
Four new integration tests and26 direct legacy regressions passed. The actual
GPU warmup used three discarded SGD updates;13 GPU checks passed for frozen
parents, strict reload, sample-hook parity and state/map sensitivity. Warm Adam
was not adopted. New weights are explicitly unqualified, with no auto-promotion.

Worker51048 and page8784 started `conditional_restorer_20261114`, budget2000,
preview10/VAL100. Live UI checks passed28/28 at150/VAL100/preview150.
Fixed256 RF old→new MAE0.036398→0.036157, NPC0.092377→0.092163 and boundary
0.096184→0.095980 changed only slightly; SSIM0.743660→0.743649 was flat.
Source preservation passed, but NPC improvement was less than0.0005, so RF
improvement and absolute quality remain false. Objects are not demonstrated
restored. Scope is native32 GT-pose rendering, not SR/autonomous gameplay.
The public entry point is `python -m turkeyworld.train_conditional_restorer`;
retain its full training checkpoint for resume, not just the format4 play export.
[GPU proof](training_runs/conditional_restorer_20261114/gpu_bundle_verification.json),
[VAL100](training_runs/conditional_restorer_20261114/validation/step_000100.json).

## H97. Expanded conditional restorer near 2M with a strict SSIM goal

The user requested about two million restorer parameters and training until
held-out RF-code/restorer SSIM is strictly greater than 0.85. The learned
247,539-parameter / 65-tensor module stopped at update 1389. Expansion preserves
that complete branch under `legacy.*`, including its parameter objects and all
65 Adam moments, options and ages. A separate `expansion.*` branch uses width
224, 8 heads and FF288 in two attention blocks. Its 1,750,259 real neural
parameters bring the trainable restorer to **1,997,798 parameters / 130 tensors**;
65 new optimizer entries start fresh. There are no padding-only parameters.

The new branch corrects the learned legacy RGB from the same quantized codes,
current state and raw material map. A zero RGB output preserves the trained
parent function exactly at activation. Both restorer branches subsequently
train. RF3 cumulative 30717 / local 1108, Base 29209, world 5906, the complete
codec and decoder/post-quant update 1725 remain frozen. Runtime receives no
Source RGB, target mask, raycast or occlusion oracle. The strict format4 bundle
uses explicit expanded architecture/config/state/hash dispatch; the old 247k
configuration and trained exports still load unchanged.

Six new architecture tests plus the combined 35 conditional tests passed,
including trained-parent RGB exactness, FP16/mode/RNG preservation, retained old
Adam state, all 65 added tensors receiving gradients after the zero output opens,
and strict old/new loading and tamper rejection. Three discarded GPU warmup
updates averaged 0.8124 seconds at 0.457 GiB peak. Actual GPU reload and migration
checks passed **14/14**. The probe's old/new Adam ages became 1392/3; production
starts from the original 1389 moments with fresh expansion states, not probe Adam.

Production worker **49844** started
`training_runs/conditional_restorer_2m_20261114`; the existing
**http://127.0.0.1:8784/** now follows this run. Health was confirmed at local
15 / cumulative restorer 1404. Initial full validation measured RF-Q/restorer
SSIM **0.7440760333** on the same pinned 256 held-out Source queries, so the
goal is not reached. Training has no fixed update budget: only
`models.rf_new.summary.ssim_mean > 0.85` on that full panel stops it. Equality,
preview scores, Source-Q reconstruction, train loss and an update count do not
substitute for the requested metric. Best full checkpoints and inference exports
retain the maximum verified RF-image SSIM. Other Source/NPC/geometry guards stay
visible; reaching this image goal would not qualify autonomous gameplay.

Each update uses 8 aligned Source queries, giving 8 RF-code and 8 Source-code
restorations with equal loss weights. Preview remains every 10 updates and full
validation every 100. The public entry point is
`python -m turkeyworld.train_conditional_restorer_2m`, without `--steps`;
resume needs the full restorer checkpoint and immutable inputs. Dynamics training
continues separately, and no game/dynamics renderer asset was promoted.

[Input snapshot receipt](training_runs/conditional_restorer_2m_20261114/inputs/snapshot_receipt.json),
[discarded GPU warmup](training_runs/conditional_restorer_2m_20261114/gpu_preflight/gpu_warmstart.json),
[strict GPU proof](training_runs/conditional_restorer_2m_20261114/gpu_bundle_verification.json),
[initial full validation](training_runs/conditional_restorer_2m_20261114/validation_initial.json).

## H98. H450 ends after ten TOTAL losses below one without a cap

The user replaced only the final dynamics-stage stopping rule. The new immutable
`rssm_moe_h450_total_loss_goal_v6` policy keeps the original 31-stage path,
100 mandatory updates at H240/H368, and the other nonfinal stages' 20-update
budget or five strict TOTAL losses below 0.5. H450 now has **no update cap** and
completes only after **10 consecutive committed optimizer updates with raw
TOTAL train loss < 1**. A loss equal to 1 or above resets its streak. The old
100-update H450 completion is disabled; no EMA, preview or validation metric
stands in for the completed full training loss.

The same-run activation at **6132 / H304 / stage update 1** changed only
`--stage-budget-plan` to `stage_budget_h450_loss_goal_plan.json`. Model weights,
named Adam, all RNG, three sampler streams, expanded-five architecture, 32×1
batching, old stage counts/streak/history and all previous policy files remained
intact. Both neural image assets stayed pinned. Ten policy tests passed,
including H450 beyond 100 updates, strict reset/ten-pass cases, resume during
the final streak and the H448→H450 threshold transition. Initial same-step proof
passed **18 checks**, with no optimizer update between snapshots; the live UI
passed 18 checks. Worker **43976** continued the same run and was confirmed
healthy at 6136. This is a training-loss objective, not a gameplay qualification.

[Immutable activation and same-step proof](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/budget_final_loss_revisions/g6132_f68598358e2d/initial_verification.json),
[v6 budget plan](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/stage_budget_h450_loss_goal_plan.json).

## H99. Seven-million-parameter restorer with preserved 2M learning

The user then requested seven million restorer parameters while retaining the
same strict RF-image SSIM > 0.85 goal. The 2M phase stopped at **local 800 /
cumulative restorer 2189**. Its complete 1,997,798-parameter / 130-tensor module
is preserved under `previous.*`, together with both Adam groups, mixed ages,
options and all old moments. An independent `expansion.*` branch uses width
384, 8 heads and FF720 in two attention blocks. Its 5,002,515 parameters and
65 fresh optimizer entries give **7,000,313 trainable parameters / 195 tensors**.
There is no padding-only capacity. A zero RGB readout preserves the trained
2M function exactly when connected; all restorer branches then train.

Inputs remain quantized codes, decoded/restored RGB32, current state20 and the
known raw 16×16 material map. No Source image, target mask, raycast or visibility
oracle enters runtime. RF3 cumulative 30717 / local 1108, Base 29209, world 5906,
E/Q/codebook and the complete decoder/post-quant update 1725 remain frozen.
Six new core tests and eight strict bundle tests passed, covering trained-parent
RGB exactness, FP16/mode/RNG preservation, all 130 old Adam entries, gradients
in every added tensor after the zero output opens, old/2M/7M loading and tamper
rejection. UI tests passed **202/202**. Actual strict GPU checks passed **14/14**;
three discarded warmup updates averaged 0.87344 seconds at 0.659 GiB peak
(0.718 GiB for the separate strict GPU proof). Probe weights/Adam were not adopted.

Worker **21756** started `training_runs/conditional_restorer_7m_20261114` and
the existing **http://127.0.0.1:8784/** now follows it. Health was verified at
**local 17 / cumulative restorer 2206**. Initial full validation preserved the
captured 2M function and measured RF-Q/restorer SSIM **0.7442107737** on the same
256 held-out Source queries. Live HTTP/served-JS/PNG checks passed **34/34** at
local 56 / validation 0 / preview 50. The goal remains unmet. Training has no hard update
budget and continues until `models.rf_new.summary.ssim_mean > 0.85`; equality,
Source-Q scores, MAE, train loss and preview measurements do not replace it.
Best checkpoints track the maximum validated RF-image SSIM. Preview remains
every 10 updates and full validation every 100; each update uses eight aligned
RF-code plus eight Source-code examples with equal weights. Other image and
geometry guards stay visible, and the goal does not qualify autonomous gameplay.

The public entry point is `python -m turkeyworld.train_conditional_restorer_7m`,
without `--steps`; resume requires its full `restorer_last.pt`, all immutable
inputs and the previous full 2M training snapshot. Dynamics continues separately
under H98. Game and dynamics renderer/SR assets were not automatically changed.

[Input/parent receipt](training_runs/conditional_restorer_7m_20261114/inputs/snapshot_receipt.json),
[discarded GPU warmup](training_runs/conditional_restorer_7m_20261114/gpu_preflight/gpu_warmstart.json),
[strict GPU proof](training_runs/conditional_restorer_7m_20261114/gpu_bundle_verification.json),
[initial full validation](training_runs/conditional_restorer_7m_20261114/validation_initial.json),
[live page proof](training_runs/conditional_restorer_7m_20261114/diagnostics/live_ui/verification.json).

## H100. Base-first geometry with five full-query experts

The user prioritized Base wall geometry, corners, occlusion and map alignment
before further RF training, then requested about 300k parameters per Base
expert. The prepared phase is `training_runs/base_geometry_20261114`, protocol
`base_expanded_geometry_v1`. Five experts retain their trained 8D Transformer
and append an independent width192 / heads6 / FF314 Transformer reading the
full144D query. A zero output192→8 preserves the trained initial function.
Each expert has **300,058 parameters**; Base grows **1,001,157→2,498,447**, adding
**1,497,290 parameters / 80 tensors**, without padding-only capacity.

All old Base tensors and 537 Adam states/six groups remain, with only the new
80 tensors starting fresh in group seven. Only Base trains. RF30717,
world5906, private opacity and the full codec/D1725/E/Q/codebook are frozen;
no restorer is attached and training/validation do not call RF. Three Source
sampler states and RNG are preserved. MAIN16, far4×0.25 and geometry8×1 retain
the fixed Source corpora, held-out256 and geometry128 panels. The initial pilot
has 2000 updates, preview10 and VAL100. Best follows raw Base SSIM, not codec
reconstruction or RF images; report-backed resume rejects NaN/future/stale best
records and retains the earliest tie. Budget completion is not a quality pass.

Seven new architecture tests plus 36 compatibility tests passed. Twelve driver
and best-resume tests passed. Three discarded GPU warmup updates passed,
averaging **1.50772 s/update**, peak **1.15978 GiB**. Production worker47280 is
running and **http://127.0.0.1:8784/** follows its healthy baseline. GPU format6
reload kept warm raw/quantized decoded RGB bit exact; independent initial CUDA
forwards measured ~1e-6 RGB numerical differences, which can change hard codes.
The production0/1/2/3/20 audit passed32 checks of old weights/moments/RNG and
sampler clocks. Initial raw Base SSIM is0.72477; no startup quality improvement
is claimed. The 7M restorer stopped at
local17277 without its .85 goal. RF-only production was not launched; it waits
for Base validation. Dynamics continues under the H98 final loss policy and
keeps its existing renderer/SR assets. No automatic game promotion occurs.

Pinned **VAL300** on the same256 Source queries measured raw Base SSIM
**0.724771957→0.728862517**, selecting best300. Frozen E/Q/D reconstruction of
Base rose **0.711908530→0.719739665**. Raw wall-boundary L1 was flat/slightly
worse, **0.095222458→0.095269307** over218 supported images; raw RGB MAE rose
0.037659558→0.038095469. The modest SSIM gains do not establish repaired
boundaries or quality, and no new image assets were promoted.
[Actual report](training_runs/base_geometry_20261114/validation/step_000300.json),
SHA `dd785ae8e656fcafe167f49e4fafc006f97b4a160e28cd1a1f1c82906b8f891d`.

[Pinned inputs](training_runs/base_geometry_20261114/inputs/snapshot_receipt.json),
[discarded GPU warmup](training_runs/base_geometry_20261114/gpu_preflight/gpu_warmstart.json),
[GPU reload](training_runs/base_geometry_20261114/gpu_bundle_verification.json),
[production audit](training_runs/base_geometry_20261114/diagnostics/independent_audit/verification.json),
[live dashboard](training_runs/base_geometry_20261114/diagnostics/live_ui/verification.json).

## H101. Learned clock and candidate spawn conditioning

After the spawn-input audit, the user requested the proposed improvement.
An opt-in `rssm_spawn_conditioning_enabled` branch is activated with
**118,736 parameters / 12 tensors**, namespace
`vjepa_backend.spawn_conditioning.*`. It is appended after the existing learned
dynamics modules. No recurrent memory is added and no old parameter is replaced.
The default remains false for legacy configurations and checkpoints.

The current own time advances conceptually to the next tick for feature
encoding only: round to30Hz, one-hot eight90-tick phases, and fraction within
the phase. Each of eight anchor candidates gets ID embedding, absolute XY,
player-relative XY/distance and known floor/separation legality. The ninth
class uses the centroid and fraction of legal non-anchor candidates. This is
a fallback-pool descriptor, not the position selected for spawning; the existing
learned256 tile head and decoder still choose its legal cell. A shared MLP
outputs nine additive logits with zero final weights. Its otherwise redundant
shared scalar bias was omitted because it cancels in nine-way CE and argmax.

The adapter is added before the existing anchor/fallback hierarchy. MoE and
CombatRSSM corrections retain their existing path. Existing Source-labelled
AUX/late spawn CE trains the new branch; the loss formula, three RSSMs/5232 memory,
batch32×1, samplers, budget and H450 final loss goal are unchanged. No Source
rank function, future image, teacher-coordinate feedback or forced placement is
introduced. Worker **45344** resumed the same run at **6412 / H450**, adding only
`--spawn-conditioning-plan` (`spawn_conditioning_plan.json`, SHA
`4d04c39a47c82a874fb891e4faf4bc4c0f9b6a488b0aeea45b44251c4807f5dd`). The progress
page remains **http://127.0.0.1:8765/**. All1587 old model tensors,430 Adam states/
six groups, RNG/three sampler streams and the complete H450 gate/history were
retained. Only the12 new tensors form fresh group seven; memory remains5232.

Six focused CPU tests passed: all eight phase boundaries±one tick including
24-second wrap, candidate-mask parity with the existing hierarchy, first-step
zero-readout/deep second-step gradients, no Source/rank calls, exact old state/
memory/spawn outputs and sampled RNG at zero activation, plus strict migration
metadata/load and tamper rejection. The preceding eight fusion regressions also
passed. This establishes safe initial behavior, not improved spawn accuracy.
The actual GPU proof passed **16/16**, independent CPU migration **28/28**,
post-flag format6/H450 compatibility **10/10**, and initial same-step resume
**26/26**. The128-case held-out spawn report is identical at zero activation,
not evidence of learned improvement. The first durable update **6413** passed
**27/27** checks: all12 new Adam states reached age1,442 total states were present,
and frozen assets remained exact. The new spawn branch received gradient
**0.029850155**, with nine actual Source respawn events. The existing policy
update counter advanced280→281 naturally; TOTAL **3.4124397** left the H450
strict-loss streak at0. Full validation remains every250 updates, next at6500.
This proves training execution, not better spawn accuracy. Base-first training
under H100 continues independently; image assets do not change automatically.

[Activation](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/spawn_conditioning_revisions/g6412_af2d46e5c2a4/start.json),
[GPU proof](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/spawn_conditioning_revisions/g6412_af2d46e5c2a4/gpu_verification.json),
[independent migration](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/spawn_conditioning_revisions/g6412_af2d46e5c2a4/independent_migration_verification.json),
[initial resume](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/spawn_conditioning_revisions/g6412_af2d46e5c2a4/initial_verification.json),
[first durable update](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/spawn_conditioning_revisions/g6412_af2d46e5c2a4/first_update_verification.json).

## H102. Ordered Base, RF3, decoder and SR training

The user requested **Base → RF3 → VQGAN D/post-quant → SR, 3000 updates each**.
Base continues from661 toward total3000, preserving its original phase and
all weights/Adam/RNG/samplers/best-history; the same-step revision passed22 checks.
Each later stage receives3000 new updates and takes the whole best checkpoint
with its matching export. Base freezes RF/D; RF freezes Base/D; D trains only
decoder/post-quant on50/50 RF-Q/Source-Q; SR refreshes its paired cache from the
selected final renderer. World5906, E/Q/codebook and private geometry stay frozen.
There is no restorer or automatic game/preview promotion.

The [ordered supervisor](turkeyworld/ordered_image_pipeline.py) is running as
**PID50340** in `training_runs/image_chain_20261114`, adopting Base45576 at1402/3000
without resetting it. Isolated GPU checks passed **RF27, decoder37 and SR19+23**;
all nine test updates were discarded. Future stages repeat their three-update
check on the actual selected parent before production. The RF check retained
617 Adam states/seven groups and exact frozen Base/D. Strict format7 RGB and
terminal-Q reload matched exactly. A loader fix preserves inherited non-unit
latent normalization; it does not change trained weights. Workers run hidden
with system-only keep-awake, so locking the screen does not require a visible
training window. Progress is at **http://127.0.0.1:8784/**; budget completion is
separate from measured quality.
The ordered-plan SHA256 is `481e93dec913e6181658b93486debb431a63c6cda021e06288236fe41f500a92`.

[Budget plan](training_runs/base_geometry_20261114/budget_3000_plan.json),
[same-step661 proof](training_runs/base_geometry_20261114/budget_revisions/g661_7cc00df3ca4f/initial_verification.json),
[launch plan](training_runs/image_chain_20261114/launch_plan.json),
[ordered plan](training_runs/image_chain_20261114/ordered_plan.json),
[RF smoke proof](training_runs/image_chain_20261114/smoke/rf3/verification.json),
[decoder proof](training_runs/image_chain_20261114/smoke/decoder/verification.json),
[SR warmup](training_runs/image_chain_20261114/smoke/sr/gpu_warmstart.json),
[SR strict reload](training_runs/image_chain_20261114/smoke/sr/verification.json).

The live8784 dashboard passed22/22 checks: Base1441/3000, preview1440,
VAL1400 and the three queued0/3000 stages match. The power guard is active:
system_required=true, display_required=false. Dynamics continues6441/H450,
batch32×1, with its original NPR/SR assets.
[Live dashboard proof](training_runs/image_chain_20261114/diagnostics/live_ui/verification.json).

## H103. Player contact repair and full-horizon gradients

The user requested repair of the large15-second player drift, first increasing
TBPTT16→64, then conditionally choosing **full TBPTT450 with MAIN16×1=16**
if it fits the GPU. The64/B32 execution was a discarded capacity probe, not
production training. The latest requested MAIN mix is **8 quiet NPC-motion,
4 no-enemy navigation,4 combat**:50%/25%/25%. The continuation is prepared from
complete **6518 / H450**. The exact new-layout GPU check passed and production
was activated after the preservation proof. A new immutable
v3 batch plan retains the complete old32×1 segment
and its source gate. New updates count16, with AUX16/late-prefix2 separately.
Player XY uses all three streams with actor weights4/16 combat,8/16 quiet
and4/16 navigation; NPC uses4/12 and8/12. The three RSSMs, spawn lineage,
existing weights/Adam and image assets stay intact. The main trajectory still
advances450 times without observation corrections; the450-step graph passes
credit across the full15 seconds without intermediate memory detaches.
Contact snapshots remain every16 ticks independently of gradient truncation.

The existing **6912 short TRAIN clips** are reused:2048 corner clips plus4864
replay clips. A new private PCG draws four additional actors per update, balanced
across free/blocked axis outcomes. Recovery horizons cycle **4/8/16/32**, starting
from the model's actual own state and RSSM memory in combat, navigation or quiet
streams. Source supplies counterfactual movement labels only during training;
future states/RGB do not correct the rollout. Circle-radius.20 wall supervision
adds .25 times the radius-normalized mean of endpoint and X-then-Y swept-path
penetration, with equal halves. These are losses, not runtime projections or
Source collision repairs. The explicitly revised MAIN percentages are separate
from contact actors, which remain separately counted auxiliary work. The new
explicit AUX16 adapter retains the saved32-sampling policy and its historical
counters, continuing actual event/anchor counts in a distinct16-continuation ledger.

The new `micro16_single_batch_revision_plan.json` and
`player_navigation_tbptt450_batch16_plan.json` pin the source checkpoint SHA,
old/new gate and batch lineage, and contact inventory. The earlier64/B32 plan
is immutable and is not reused for450/B16. All old weights,442 Adam states in
seven groups, global RNG and old private sampler states must be conserved.
The original H450 goal remains **raw TOTAL<1 for ten consecutive updates**,
with no update cap or EMA normalization. Activation requires the existing
streak0 boundary. Because the objective adds quiet-player/contact/wall terms,
its TOTAL has an explicit revision tag and is **not directly comparable with
historical TOTAL**; source metrics remain archived rather than reinterpreted.

A fresh actual autonomous450 baseline is required before new SGD. Validation
continues every250 updates and reports player drift at1/4/8/16/32/64/128/256/450,
circle/path wall intersections and short free/blocked-axis errors. A separate
physical selector keeps whole model+Adam checkpoints by fixed held-out drift
and collision metrics; lower train loss does not determine the selection, and
no automatic game deployment occurs. The original protocol/merge and replay/batch
regressions passed; additional tests cover450/B16 lineage, historical sampler
quanta and the read-only proof. No drift improvement is claimed.

The actual RTX4090 capacity checks all completed real backward, finite gradient
clipping and Adam steps; their weights were discarded:

| Gradient span / MAIN batch | Peak allocated GiB | One optimizer update, seconds |
|---|---:|---:|
| TBPTT64 / B32 | 4.25884 | 133.7512 |
| TBPTT250 / B32 | 11.13769 | 133.7980 |
| **TBPTT450 / B16,4 combat/8 quiet/4 NAV** | **9.00306** | **197.0446** |

The selected450/B16 check passed **11/11**, including all three actual450-step
long-stream backpropagations and contact backward.434 parameter tensors received
gradients,8 remained untouched. Peak reserved memory was9.25391GiB; total probe
wall time with setup was257.2538 seconds. Source6518 SHA256
`69b1793d58837499c0910476df6d3c5707f840ecf7053bcbd66cdad8b9eedfbe`
remained unchanged, with no probe weight promotion. The250/B32 and450/B16
measurements use different batch sizes, actor proportions and auxiliary sizes,
so they are not a controlled timing comparison. Capacity success establishes
that this update executes, not that navigation drift has improved.

Production **PID52644** started from6518 at **2026-11-14 11:34:14UTC /
14:34:14MSK**. The initial preservation proof passed **28/28**, comparing all
**1599 model tensors,442 Adam states/seven groups**, existing RNG/private samplers
and the old gate preserved inside the explicit v3 batch lineage. The initial
child checkpoint SHA256 is
`8b73e59cb68b98d0f19cf3c44b54325d31869db54bbed1e45c51e1afca2c6177`;
the full-horizon plan SHA256 is
`42fc3871139ae40138ae542aa59a2ba62b3a579a513932e7238452c655f87d03`.
The fresh450 baseline completed before new SGD:48 trajectories/21600 transitions,
player final drift mean1.81483/p958.12678 tiles,4920 player wall-overlap steps.
This is the initial6518 quality, before new training. Production update **6519**
is saved: all three streams actually use TBPTT450, MAIN4combat/8quiet/4NAV,
TOTAL4.478004,107.69 seconds. Its independent first-update proof passed:
434 Adam states advanced once,8 stayed untouched,1156 frozen tensors remained
bitexact; NAV/quiet/contact advanced by exactly4/8/4 draws.
Drift improvement after the new training has not yet been measured.

[Fresh6518 baseline](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/validation/step_006518/player_navigation_validation.json),
[production6519 proof](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/player_navigation_revisions/g6518_69b1793d5883/tbptt450_batch16/first_update_verification.json).

[Production launch](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/player_navigation_revisions/g6518_69b1793d5883/tbptt450_batch16/start.json),
[initial28/28 preservation](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/player_navigation_revisions/g6518_69b1793d5883/tbptt450_batch16/initial_verification.json),
[64/B32 discarded probe](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/player_navigation_revisions/g6518_69b1793d5883/gpu_probe.json),
[250/B32 capacity](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/player_navigation_revisions/g6518_69b1793d5883/capacity_250_b32.json),
[450/B16 exact50%/25%/25% capacity](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/player_navigation_revisions/g6518_69b1793d5883/capacity_450_b16_enemy50_nav25_combat25.json).

[Revision protocol](turkeyworld/rssm_player_navigation_revision.py),
[contact replay](turkeyworld/rssm_player_contact_replay.py),
[wall objective](turkeyworld/rssm_player_geometry_objectives.py),
[physical validation](turkeyworld/rssm_player_navigation_validation.py),
[read-only boundary proof](training_runs/diagnostic_tools/verify_player_navigation_revision_20261114.py).

## H104. Own-state local player and enemy motion

The user requested free-axis, invalid-NPC and enemy-speed fixes after the
6519–6632 audit. **H450/TBPTT450, MAIN16=4combat+8quiet+4NAV**, all weights/Adam/
RNG/samplers and **TOTAL strictly<1 ten times consecutively** remain. The
cancelled0.3 threshold was never applied.

`rssm_player_enemy_local_motion_repair_v3` supplies exact one-movement-step
Player Source labels even with an invalid NPC, provided the live player body
is legal. NPCs are disabled only in the label query; NN state/hz are not
corrected. Free/blocked/partial masses2/1/1 and free-progress loss counter false
stops. Up to two contact branches reserve nonzero quiet prefixes; shortfalls
are explicit. Direct labels follow neural rollouts.

NPC own-memory H4/8/16/32 branches prefer2combat/2quiet and initial/nonzero
prefixes. Own-aligned local supervision is weighted
0.5×(vector+speed+0.25direction+0.1own-prefixXY). NPC circle/swept X→Y loss uses
radius0.25/weight0.25/radius² and Source lifetime masks retaining ownmissing.
Respawn motion skips swept/localvelocity only; global SourceXY is retained.
Recorded quiet velocity/direction are diagnostic. Local horizons use a16-update
Latin-square schedule independent of latentcycle. Gameplay receives no Source correction.

Source **6647**, SHA256 `fa36be6385f3cd9d1b97d57b0a1498fa4563b4e3ca4e083293928a092e806f82`.
Plan SHA256 `0eeee76f7c9fc8fd13cf2748db27d04298da8cb6ded2a660b13ab14151ab4902`.
Discarded GPU15/15:132.44s/update,9.22085GiB allocated/9.45703 reserved.
Production **PID54052**,2026-11-14 **19:07:59MSK**. Initial22/22 preservation
verified1599 tensor entries,442 Adam states/seven groups,1156 frozen entries
and all old clocks. Fresh450 baseline completed before SGD; first production
update **6648** passed31/31. New TOTAL has a separate curve and is not directly
comparable with the former objective. Validation250/preview10 remain; quality
improvement is not yet claimed.

[Plan](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/motion_correction_plan.json),
[GPU](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/motion_correction_revisions/g6647_fa36be6385f3/gpu_probe.json),
[preservation](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/motion_correction_revisions/g6647_fa36be6385f3/initial_verification.json),
[first update](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/motion_correction_revisions/g6647_fa36be6385f3/first_verification.json).

## H105. RGB-only learned enemy SR with a frozen background

The user requested that SR reconstruct enemy appearance neurally instead of
replacing a silhouette with fixed sprite pixels. A separate3000-update trial
was implemented as `neural_enemy_restoration_sr`: frozen Semantic EDSR plus the
copied five-expert RGB residual form the background, while a new full-frame
U-Net/PixelShuffle decoder predicts native128 RGB and alpha from RGB32 only.
There are **2,162,020 trainable parameters /58 parameter tensors** and
**2,010,828 frozen parameters**. Source masks cannot hide RGB errors through the
predicted alpha: appearance and opacity have separate direct losses. Source
RGB, masks, distance and animation are labels only; inference needs no GT,
RSSM, map, sprite, atlas buffers or atlas file. Atlas is used only by the
separate frozen reference arm; both arms receive matching RGB inputs without
a hypothesis hint.

The parent is the completed image-chain **BEST FULL SR, local2600/global110705**,
SHA256 `dea6376208b1ee3f72202df4e0f9d7d76b1499807730a1dccb5454b376dd8f79`.
All183 copied background tensors are bit exact; a CPU check measured maximum
background difference0 against the old pre-composite RGB. The new standalone
checkpoint contains no atlas pixels. Its optimizer is fresh Adam for the new
58 tensors; no reuse of the old Atlas Adam is claimed. The actual final RF3/Q/
decoder cache supplies **8192 TRAIN/1024 VAL** frames without a hypothesis
sidecar. Sampling balances HR-visible/empty frames, near/mid/far and available
hit/death animation phases. Validation uses the parent's exact256 held-out
indices, with a separate data/panel fingerprint. Preview15, validation100,
batch8 and full atomic model/Adam/sampler/RNG saves after every update are used.

The first disposable CLI startup stalled before a plan or SGD was produced.
Only that isolated probe was cancelled. Initializing CUDA and setting two CPU
threads before the trainer/keep-awake work allowed the replacement GPU trial
to complete **three discarded updates and9/9 checks**, including all58 Adam
states at step3,24 sampler draws, unchanged frozen background, no atlas in the
new model and finite standalone reload. The CLI now establishes that runtime
order before KeepAwake/preparation. After initial validation, its stage label
switches to actual decoder training; the live UI also corrects the earlier
stale initial-validation label.

Production **PID4636** runs in `training_runs/sr_neural_enemy_20261114`, monitored
at **http://127.0.0.1:8786/**. Plan SHA256
`4fb82988acfd45c3e75dc636fc506162ef90171fabdaad15260cd8bba8ec85df`.
Dynamics continues independently under H104; no automatic game promotion occurs.

The **step200** same-panel comparison is an early snapshot, not completion:

| Metric | Learned RGB/alpha | Frozen Atlas, RGB-only input |
|---|---:|---:|
| Whole-image SSIM |0.78133|0.77747|
| Whole-image MAE, normalized RGB |0.03669|0.03361|
| Foreground MAE,205 visible frames |0.12983|0.15095|
| Visible alpha IoU |0.09934|0.31895|
| Empty soft-mask leakage,51 frames |40/51|0/51|

The leakage check asks whether **any** predicted alpha pixel exceeds0.05;
78.4% does not mean that78.4% of empty images contain a fully painted enemy.
Foreground reconstruction and SSIM improved in this snapshot, while overall
MAE and mask precision/overlap still need work. SSIM>0.91 and MAE255<2.5 have
not passed, nor has gameplay quality. No conclusion of final success is drawn
from the frozen weapon/background or the3000-update budget.

[Plan](training_runs/sr_neural_enemy_20261114/training_plan.json),
[GPU verification](training_runs/sr_neural_enemy_20261114/gpu_probe_verification.json),
[step200](training_runs/sr_neural_enemy_20261114/validation/step_000200.json),
[model](turkeyworld/neural_enemy_restoration_sr.py),
[trainer](turkeyworld/train_neural_enemy_restoration_sr.py).

## H106. Foreground-only SR correction with complete-state continuation

The user requested a fix for the learned enemy layer's colored patches. V1
was stopped at a complete optimizer boundary at **absolute step2308**; its
original directory and loss history remain intact. A four-case forward check
showed that the frozen background already depicted recognizable enemies, while
the newly learned RGBA layer covered them with brown paint. Near/mid/far and
empty examples had maximum alpha approximately0.9989/0.9975/0.9932/0.9879.
This identifies excess paint by the learned layer; it does not establish the
quality of the new correction. The frozen wall blur remains a separate limit.

The new explicit loss protocol is
`neural_enemy_restoration_foreground_v2`; V1 stays available unchanged. Source
foreground RGB, boundaries and SSIM directly supervise raw RGB and the composite.
Both sides use the same detached frozen background outside Source foreground,
so background color errors cannot reward the enemy head for painting an opaque
patch. Alpha has separate positive/negative BCE and Dice, plus hard-negative
BCE on the **worst32 actual background pixels per image**. Full and hard-pixel
background preservation and empty-frame image-change penalties remain active.
Predicted opacity cannot hide the direct appearance loss. Whole-Source-image
errors are retained as diagnostics, not V2 background-reconstruction gradients.
The architecture, frozen background, RGB-only inference and no-sprite contract
remain unchanged; Source masks are loss/validation labels only.

The correction uses a separate run,
`training_runs/sr_neural_enemy_corrected_20261114`, with **3000 additional updates**,
absolute **2308→5308**, batch8, phase preview15 and phase validation100. The source
checkpoint is pinned at SHA256
`3c9731cf5450da1d312d33bef718ad192770e3c5af5d74eb608bd23f0d963de5`;
its source plan is
`4fb82988acfd45c3e75dc636fc506162ef90171fabdaad15260cd8bba8ec85df`.
All weights, **58 Adam states**, process RNG and private sampler are continued,
with no optimizer reset. Source sampler draws are18464, not0. The initial full
checkpoint is written before the new baseline validation. The old best remains
archived in `resume_source.previous_best`; it is not compared to a V2 loss. A
new best ledger begins from the V2 baseline on the same source weights and
same256 held-out images from the8192 TRAIN/1024 VAL cache.

The dashboard at **http://127.0.0.1:8786/** distinguishes phase progress0→3000
from the absolute step, rejects old-protocol training rows and shows the source
baseline only as validation. Validation checks protocol, source/panel identity
and matched Atlas reference. Empty-mask diagnostics now distinguish any soft
pixel≥0.05 from dense pixels≥0.5, with dense pixel count, maximum/top64 alpha
and top64 image change relative to the frozen background. These are leakage
measurements with explicit empty-frame support, not a count of complete enemies.

CPU checks passed **34 restoration tests,9 trainer tests and238 dashboard tests**;
the dashboard Worker build also passed. A separate three-update discarded GPU
probe reached absolute2311 and passed **21/21 complete-checkpoint checks**:
initial weights, all58 Adam states, RNG and sampler matched source2308; only
the learned head changed on the first update, while all183 frozen background
tensors stayed bit exact. Production **PID6004** then started the new3000-update
phase; its own first update also passed21/21 continuation checks. The dashboard
pointer was switched only after the new baseline and first-update proof passed.

On the same256 validation images, the **first100 correction updates** measured:

| Metric | Source2308, V2 baseline | Absolute2408, phase100 |
|---|---:|---:|
| V2 validation loss |1.54835|0.98976|
| Whole-image SSIM |0.77939|0.78634|
| Foreground MAE,205 visible frames |0.13501|0.12691|
| Enemy ROI SSIM |0.21472|0.29744|
| Mask precision / recall |0.70463 /0.79605|0.93422 /0.54477|
| Visible alpha IoU |0.44161|0.25201|
| Empty frames with any alpha≥0.05 |28/51|26/51|
| Empty frames with any alpha≥0.5 |21/51|1/51|
| Dense alpha pixels across51 empty frames |5104|7|
| Raw RGB foreground MAE |0.14111|0.14239|

The next two checks give a more complete picture of the same held-out panel:

| Metric | Source2308 | Phase100 | Phase200, absolute2508 | Phase300, absolute2608 |
|---|---:|---:|---:|---:|
| Whole-image SSIM |0.77939|0.78634|0.78697|0.78439|
| Foreground MAE |0.13501|0.12691|0.12644|0.12855|
| Enemy ROI SSIM |0.21472|0.29744|0.29883|0.28928|
| Raw RGB foreground SSIM |0.30223|0.29687|0.30181|0.30395|
| Mask recall |0.79605|0.54477|0.51968|0.59127|
| Far-enemy mask recall |0.90856|0.02876|0.12267|0.17655|
| Dense empty frames |21/51|1/51|1/51|2/51|
| Dense alpha pixels across51 empty frames |5104|7|12|73|

False opaque paint remains substantially reduced across all three checks.
Enemy reconstruction is incomplete: far-enemy recall partially recovers after
phase100, but remains far below source, and whole-image SSIM is not monotonic.
At300, raw RGB foreground SSIM slightly exceeds source while visible-mask
recall remains lower. These are early trade-offs, not proof that the enemy is
accurately reconstructed. The image-quality and gameplay goals have not passed;
there is no automatic promotion to the game.

[Pinned source receipt](training_runs/sr_neural_enemy_corrected_20261114/source_receipt.json),
[four-case diagnosis](training_runs/sr_neural_enemy_corrected_20261114/frozen_background_diagnosis.png),
[production continuation proof](training_runs/sr_neural_enemy_corrected_20261114/continuation_verification.json),
[V2 baseline](training_runs/sr_neural_enemy_corrected_20261114/validation/step_002308.json),
[phase100 validation](training_runs/sr_neural_enemy_corrected_20261114/validation/step_002408.json),
[phase200 validation](training_runs/sr_neural_enemy_corrected_20261114/validation/step_002508.json),
[phase300 validation](training_runs/sr_neural_enemy_corrected_20261114/validation/step_002608.json),
[objective](turkeyworld/neural_enemy_restoration_objectives.py),
[continuation checks](turkeyworld/neural_enemy_restoration_continuation.py),
[validation](turkeyworld/neural_enemy_restoration_validation.py).

## H107. SR V3: details, animations and matched background validation

V2 completed all3000 additional updates at absolute5308. Its loss-best is also
the final checkpoint: SSIM0.78651, foreground MAE0.12517, ROI SSIM0.30196,
raw RGB foreground SSIM0.33562, mask precision/recall0.93027/0.58807. Relative
to source2308, dense false alpha pixels on51 empty images fell5104→321 and
dense-error frames21→4, while recall fell from0.79605. This fixed much of the
paint but did not complete enemy reconstruction or the image/gameplay goals.

The user authorized targeted continued training. V3 uses
`neural_enemy_restoration_detail_v3`, **PID56288**, in
`training_runs/sr_neural_enemy_detail_20261114`, at **http://127.0.0.1:8786/**.
Its3000-update phase continues **5308→8308**, batch8, preview15 and validation100.
Source SHA256 is
`48c35544bc473d8ff1a8c5d30c7bec72038ce823cdfe917d0b4819c1ee620b69`;
plan SHA256 is
`ad5458ce22826d140c50c38c15a2027a51987b19e6b5a0661b2f9f970a329082`.
The same58 Adam states, all weights, sampler and process RNG continue without
reset; the new model still takes only RGB32, with the same2,162,020 trainable
parameters and2,010,828 frozen background parameters. No new architecture,
Atlas pixels, RSSM input, GAN or Source inference hint was added.

An actual no-SGD gradient audit at source5308 measured excessive mask pressure
relative to RGB. Fixed group scales are **mask0.5/direct RGB4**, with composite
foreground and preservation groups unchanged. The death-cohort mask/RGB trunk
gradient-norm ratio changed from approximately18.2 to2.63 in the audit. These
are measured training gradients, not image-quality improvements. V3 also adds
positive-alpha consistency, hard-positive alpha BCE, the worst32 foreground RGB
pixels, and direct RGB/edge/SSIM supervision on aspect-preserving native/32/64
enemy crops. Predicted alpha still cannot hide raw RGB errors; both comparison
sides use the same detached frozen background outside Source foreground.
Weights are fixed, with no EMA normalization and no Source-background painting
gradient. V1/V2 loss implementations and historical curves remain separate.

The main held-out panel remains the same256 images from8192 TRAIN/1024 VAL.
A CPU audit found845 visible/179 empty VAL images, including28 attacks,
184 hurt,20 early deaths and40 late deaths. The additional fixed panel uses
**120 unique VAL images: all28 attacks,32 hurt, all20 early deaths and all40
late deaths**. Its fingerprint is
`f937c3dd727afa8016eaa95d78a6a22af748d5f0dfceaa82e14e8fd9a0ccb146`.
These labels are never used as TRAIN. Each death phase has only5 far cases,
so its far-distance support is reported explicitly rather than treated as
robust generalization. Additional-panel scores do not alter the main256 loss.

V3 adds a matched **frozen-background-only** arm directly from the same forward
output, without an additional network call. Preview retains Source/NPR/Atlas/
learned output and appends the frozen background as its fifth row. The background
already depicts some enemies; its alpha/detector metrics are unavailable, not
evidence of a perfect detector. A separate guarded visual-best ledger averages
near/mid/far raw RGB SSIM, ROI SSIM and mask F1. A candidate cannot worsen the
source's dense-empty counts/pixels, raw RGB SSIM, ROI SSIM or recall; precision
has0.03 tolerance. This selection is separate from loss-best and is not gameplay
qualification or automatic deployment.

Both the three-update discarded GPU probe and the production continuation
passed **21/21 conservation checks**. CPU objectives25/25, validation11/11 and
the combined dashboard246/246 tests passed. The first same-panel readings are:

| Metric | V3 source5308 | Phase100,5408 | Phase200,5508 | Frozen background only |
|---|---:|---:|---:|---:|
| Whole-image SSIM |0.78651|0.78515|0.78632|0.79109|
| Foreground MAE |0.12517|0.12753|0.12696|0.12483|
| Enemy ROI SSIM |0.30196|0.29662|0.30017|0.33675|
| Raw RGB foreground SSIM |0.33562|0.33777|0.33720|—|
| Mask precision / recall |0.93027 /0.58807|0.88824 /0.66095|0.89369 /0.61331|—|
| Dense-empty frames / pixels |4/51 /321|3/51 /395|4/51 /545|—|

Raw appearance improves slightly and recall increases, while the composite
has not improved and still trails the actual frozen background. The guarded
visual-best remains source5308 at this snapshot. This ongoing phase is not a
quality success; its formula is kept stable while evidence accumulates.

[Plan](training_runs/sr_neural_enemy_detail_20261114/training_plan.json),
[gradient audit](training_runs/sr_neural_enemy_corrected_20261114/gradient_attribution_5308_detail_v3.json),
[continuation proof](training_runs/sr_neural_enemy_detail_20261114/continuation_verification.json),
[baseline](training_runs/sr_neural_enemy_detail_20261114/validation/step_005308.json),
[phase100](training_runs/sr_neural_enemy_detail_20261114/validation/step_005408.json),
[phase200](training_runs/sr_neural_enemy_detail_20261114/validation/step_005508.json),
[visual selection](training_runs/sr_neural_enemy_detail_20261114/visual_best.json).

## H108. MAIN18 full-BPTT ladder with a strict TOTAL loss gate

The user requested shorter MAIN stages with full credit through each current
trajectory, then changed the effective batch to **18×1=18**, evenly divided
into **6 combat,6 quiet enemy-motion and6 no-enemy navigation actors**. The prior
local-motion run was stopped at the complete optimizer boundary **6686**. Its
full source is pinned at SHA256
`d02888a4318b92bae08cd103336ea018522f1a7045a755bfc58c8f6b1962c6f6`.
The prepared plan SHA256 is
`290d86e4ad6c85b1bd028b2190befb725d7acaa3e145245e27f6597751665480`.
The old active worker PID54052 stopped. After GPU checks, production worker
PID47660 launched through the early-CUDA bootstrap. A genuine physical450
baseline at6686 completed before the first real SGD6687, which trained
H16/full-BPTT16 with actual MAIN18=6/6/6.

The new immutable `rssm_main18_full_bptt_loss_ladder_v1` plan preserves the old
player, motion, batch and policy plans as exact source history. A fresh stage
clock starts at **16,48,80,112,144,176,208,240,272,304,336,368,400,432,450**.
Every stage advances only after **raw TOTAL strictly<1 on10 consecutive complete
optimizer updates**; equality resets its streak. There is **no stage update cap**,
no mandatory long update and no reuse of earlier stages' qualifying streaks.
Source-support checks, EMA, component losses and validation are not substitutes
for the raw TOTAL criterion. The final H450 criterion remains<1×10 and terminates
only after its own ten qualifying updates, without claiming gameplay quality.

MAIN Player/NPC/Combat graphs retain memory and integrated coordinate gradients
through **all current H transitions**, with a single BPTT chunk H, from16 ticks
(0.53seconds) to450 (15seconds). Player/router actor weights become1/3 per stream;
NPC actor means use6/12 combat and6/12 quiet. Historical B16 exposure is retained
verbatim; new exposure counts18 actors and18×H transitions per completed update.
Navigation and quiet samplers continue their private PCGs with6 new draws each;
combat makes18 episode draws, keeps6 and records12 discarded draws. AUX16,
late-prefix2, contact4 and enemy-local4 stay separate. Existing local Latin-square
horizons are capped at the current MAIN H, so H16 does not hide an H32 gradient
branch. Unavailable long own-prefix quotas remain reported as shortfalls.

All weights,442 Adam states/seven named groups, original sampler states/RNG and
frozen renderer assets continued without reset. The new formula
and MAIN mix have a separate loss identity; old totals are historical. A fresh
physical baseline precedes SGD; independent physical validation stays450 steps
every250 updates, while image/map preview shows the current training H. Resuming
requires the same new plan via `--main-ladder-plan` plus all prior immutable plans.

CPU validation passed **156 core tests**, including genuine full-memory gradient
checks and the complete15-stage qualifying path; **9 operational guard tests**
also passed. The dashboard's combined246 tests/build passed. Actual GPU checks
then passed: the complete three-stream H450/MAIN18/full-BPTT capacity update
used **7.358GiB** peak allocation and **208.512seconds**. This disposable probe
did not become a production H450 commit. Production initial conservation
passed **29/29** checks, and its first complete H16 update passed **38/38**.
The active training loop starts H16 and advances only by its uncapped<1×10 gate;
the old H450/B16 capacity result was not substituted for this B18 measurement.

[New plan](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/main_ladder_plan.json),
[source receipt](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/main_ladder_revisions/g6686_d02888a4318b/source_receipt.json),
[curriculum](turkeyworld/rssm_main_ladder_revision.py),
[conservation verifier](training_runs/diagnostic_tools/verify_main_ladder_20261114.py),
[guarded operations](training_runs/diagnostic_tools/activate_main_ladder_20261114.py).

The unactivated draft plan's copied denominator label was corrected from16 to18
without changing numeric actor weights. A second preparation check exposed
integer anchor-count keys becoming strings in JSON. The new plan/digest now
canonicalize JSON, while raw source sampler dictionaries remain typed in the
pickle checkpoint; resume validates old contracts using that raw snapshot.
An integer-key/PCG pickle-resume test passes. Unused drafts are archived rather
than rewriting any activated historical plan.

## H109. Preferred frozen SR background with a new RGB detail branch

After V3 completed at **8308**, the user preferred its bottom, frozen-background
row and requested added detail. On the same256-frame panel, V3 composite SSIM
was **0.78724546**, versus **0.79109207** for that exact background. The base is
unchanged throughout V3; pinned source5308 therefore supplies the same preferred
weights, SHA256 `d22e704e6a919f362deb9b5d37bf86c03931c89e27745a9165cc6af97a6e39d3`.
The completed V3 optimizer/run is preserved; a later stop marker had no training
effect because the run had already finished.

`background_detail_sr_v1` creates a separate architecture: **2,010,828 frozen
base parameters** plus **526,131 new RGB residual parameters/42 tensors**.
The copied Semantic/EDSR/RGB-MoE base is exact. Its new zero output head makes
initial display equal the preferred base bit for bit; there is no alpha gate,
sprite insertion, old RGB/alpha head, Atlas asset or RSSM inference input.
The bounded `0.25*tanh` correction can restore color/shape and detail; soft
low-frequency/base and residual penalties discourage broad repainting, without
forcing a high-pass-only image. Source foreground/background/edge/SSIM targets
are training labels only. Background edges include sky and gun, not just walls.

This isolated `training_runs/sr_background_detail_20261114` experiment has fresh
Adam for the new42 tensors and a fresh private sampler, **local0→3000**, batch8.
Source5308 is provenance, not reused Adam history or the new optimizer clock.
The same8192 TRAIN/1024 VAL paired NPR32→Source128 corpus,256-frame main panel
and120-frame animation receipt remain pinned; the latter covers28 attack,
32 hurt,20 early-death and40 late-death visible frames. Both measured arms come
from a single forward: preferred base and base with learned details.
Preview every15 shows four Russian-labelled rows: Source/NPR/preferred base/new
result. Full validation every100 measures NPC/background errors, contours,
ROI SSIM, background edges, low-frequency change and localized empty-frame
paint. Alpha/Atlas metrics are absent rather than invented zeroes.

The visual selector accepts only nondecreasing overall/visible-enemy ROI SSIM
and nonincreasing Source MAE on empty frames versus the exact preferred base.
It ranks eligible weights by mean overall/foreground-ROI SSIM, separately from
objective-best. Missing support is not a pass. No automatic gameplay promotion
or quality claim follows the3000-update budget. The separate SSIM>0.91 and
MAE255<2.5 image target still needs measurement.

The immutable new plan SHA256 is
`63111f0aeca33c54532aa0902c251086b361c22691c7a6c18883342e33430294`;
its preparation verified the copied preferred base and exact zero initial output.
The21 background-detail CPU tests and250 dashboard tests/build passed.
The actual three-update GPU probe passed **22/22** checks. Production
**PID53936** started at **18:42:45UTC**, with **20/20** preservation checks;
http://127.0.0.1:8786/ now serves this run. Frozen base tensors remain bit exact.
The first new weights matched replay within1e-7, maximum difference1.87865e-8
in one weight; source indices, loss and samplers matched. This floating-point
tolerance is stated separately from the base's exact preservation.

At VAL1400, overall SSIM is **0.79117162** versus base **0.79109207**;
enemy ROI SSIM is **0.33629846** versus **0.33674788**. Mean image correction
is only0.00057081 on[0,1], while base Source MAE is0.03074407. This is a small
pixel-error change, not demonstrated recovery of enemy detail. The visual guard
retains the baseline; no image/gameplay qualification or promotion is claimed.
A separate read-only gradient audit is examining whether preservation terms
anchor the residual near identity, without changing the active training code.
Existing V1/V2/V3 and MAIN18 UI contracts remain covered. See
[GPU proof](training_runs/sr_background_detail_20261114/gpu_probe.json),
[production proof](training_runs/sr_background_detail_20261114/production_verification.json),
[plan](training_runs/sr_background_detail_20261114/training_plan.json),
[model](turkeyworld/background_detail_sr.py),
[loss](turkeyworld/background_detail_objectives.py),
[held-out comparison](turkeyworld/background_detail_sr_validation.py),
[trainer](turkeyworld/train_background_detail_sr.py).

## H110. Frozen current SR, new private preview assets and continued MAIN18

The user requested freezing the current upscaler, using it with the latest
renderer in dynamics' “What the model shows”, opening the comparison game,
and then continuing dynamics. The background-detail SR had already completed
all **3000** updates. Its current full checkpoint was copied read-only to
[`sr_frozen_step_003000.pt`](training_runs/sr_background_detail_20261114/frozen/sr_frozen_step_003000.pt),
SHA256 `c7912165e80d555f9b09c521a3d0b6c46ac3915d1ff879fbdd35d815385dc388`.
The [freeze manifest](training_runs/sr_background_detail_20261114/frozen/freeze_manifest.json)
records current-last selection, source5308,526131 new parameters and2010828 frozen
base parameters. All SR workers are stopped/completed. This explicit user selection
does not override the measured quality conclusion: final overall SSIM0.79118875
barely exceeds base0.79109207, while enemy ROI0.33601918 is below base0.33674788;
the automatic visual-best still retains the initial base.

Dynamics stopped at the complete **6745** boundary, SHA256
`3fa021fa458d8be2adaeae4991e7bb715fa345e0cf0400d526d871b340a11e7d`.
Its MAIN18 ladder retained source6686,59 completed updates, H16/stage59 and streak0.
The new explicit direct-child preview revision v5 selects the final format8
**RF3/NPR32317 + Base31709 + decoder4125**, renderer SHA256
`40f9373125c038a200b4386c25a375fa2c32d6902447a9f8b2b8cfa1e70985b6`,
and the frozen **background-detail SR3000**. It has no restorer, alpha or Atlas
replacement. The selected-plan SHA256 is
`cd1d32d868255245471908221b0f634595b12a20ee65f2128f5e2d6ccf5e5492`;
the v5 revision SHA256 is
`ddf8f9e4ddced88b0af58d3ccb1dd4153d810b7a04108522f23aa2b314c0c5d2`.
Previous asset plans and the MAIN18 plan/source genealogy remain unchanged.

An explicit RF3 image profile builds a fresh **raw training-model clone** and
loads the complete image chain once. Original Combat/RSSM renderer-lineage
validators remain strict; the code does not load a new chain onto an already
modified old preview clone. Current dynamics weights are copied into that
private clone on each preview. Source frames/states remain targets only.
The actual preview6745 contains the same current-H16 trajectory in five rows,
neural SR and bilinear output, plus its map drift. Its caption identifies
**3 Euler steps**, NPR32317/Base31709/decoder4125/SR3000.

The asset-only resume passed **25/25** same-step checks at6745: all1599 model
tensors,442 Adam states/seven groups, three serialized RNG streams, complete
typed training/sampler ledgers and MAIN18 stage/streak counters were preserved
before SGD. No physical450 baseline was repeated, no H or streak reset, and
no training image assets or old plans were replaced. Worker **PID53264** continued
training; its first complete **6746** update retained H16 with stage60/streak0,
the same18×1=18 and6/6/6 actor mix, and the original uncapped TOTAL<1×10 gate.

The Source/neural comparison window **PID47036** actually launched at
**19:09:43UTC**, using the same RF3 and frozen SR, with1024 neural display.
The user closed it after about177seconds. The last trial recorded44 simulated
transitions and the session had5 resets, including T combat switching; this was
not a successful autonomous450-step test. It was not reopened. Dynamics training
continues; image/gameplay quality is not declared passed.

[v5 revision](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/preview_assets_revision_npr32317_decoder4125_sr_detail3000_v5.json),
[same-step preservation proof](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/preview_asset_revisions/g6745_3fa021fa458d/image_chain_resume_verification.json),
[game smoke](training_runs/play_frozen_bg_rf3_g6745_20261114/runtime_smoke_verification.json),
[game report](training_runs/play_frozen_bg_rf3_g6745_20261114/report.json).

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
The latest conditional RSSM continuation uses main-microbatch16, accumulation1,
effective16; separate Source auxiliary clips
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

## H111. Dreamer-style actor–critic inside the frozen learned environment

The user requested an actor–critic that learns inside the learned world. A new
isolated learner, `training_runs/dreamer_actor_critic_20261114`, was implemented
and actually started as **PID6176**, initially pinning world step **6852**. Its
policy starts with **100 reward/continuation warmup updates**. This is an
activation record, not a claim that the policy has learned navigation or combat.

After58 completed learner updates, a routine same-plan checkpoint continuation
started **PID22856**. Its preservation receipt passed **9/9 checks**: policy and
reward-head tensors, named Adam histories, Torch/CUDA RNG, both replay samplers
and Python RNG were retained exactly; the pinned world stayed at6852. The initial
PID6176 is historical. No policy counter or warmup clock was restarted.

The saved production history subsequently records the first actor/critic update
at **101**, after the100-update head warmup, on the same world6852. The retained
Source report at **100** uses four starts and32 transitions; it precedes that
first actor update and is a baseline, not evidence of policy improvement.
Its separate head panel still missed **18/18 real deaths**, with0 false deaths
among558 alive window measurements. Frozen-model checks passed and gameplay
qualification remained false. A real Edge page capture at128 confirms the UI
displayed actor/critic learning, without equating those later losses with VAL100.

The final observed activation snapshot was **126 learner updates/26 actor
updates**, PID22856 running on world6852. VAL100 Source returns were learned0.58,
random0.58 and scripted0.60; this pre-actor baseline establishes no learned-policy
advantage. Its head reward MAE was0.03676786 and continuation Brier0.06505723,
despite the0/18 death recall. A separate discarded refresh probe passed **8/8**
checks for world6852→6862 without SGD or changes to policy/reward-head tensors.
No online Source collector is implemented: policy learning currently benefits
the manual game indirectly through diagnostics, rather than taking its controls
or automatically adding new training episodes.

The user then explicitly requested stopping the policy learner. It stopped
gracefully at **200 learner updates/100 actor updates**, after completing its
validation boundary; native status is `stopped` and PID22856 has exited. The full
checkpoint200 was retained, SHA256
`bf59f0e0ea177abd55b01b54f828450db463ad5fc24d1fe9f83836668c03c45b`.
The policy remains stopped. Main dynamics PID53264 continued running at observed
step6869 with its existing training settings. See the
[user-stop receipt](training_runs/dreamer_actor_critic_20261114/user_stop_receipt.json).

The main dynamics worker **PID53264** continues its existing MAIN18 curriculum:
six navigation, six quiet-enemy and six combat actors, full current-H BPTT and
the uncapped TOTAL<1 ten-consecutive-update gate. Its validation interval remains
250. The policy learner freezes a separate complete world snapshot and all image
models; it neither writes gradients into that snapshot nor controls the user's
manual game. Policy imagination preserves all three RSSMs' prior h/z memories
and predicted state20, without future Source frames or engine corrections.
Each policy update starts from **18 MAIN replay seeds,6/6/6**, and imagines
**15 transitions**. The actor factorizes movement, strafe and turn into three
choices each, plus binary fire:54 possible joint actions from11 logits.

Reward and continuation heads learn from recorded Source TRAIN transitions.
Their objective mixes **80% MAIN replay and20% separate short-event replay**;
the short stream has its own saved RNG and supplies real player-death examples
without changing the18 policy-start distribution. The existing short corpus
contains **73 TRAIN deaths and18 held-out VAL deaths**, all genuine HP0.08→0
transitions, with separate seeds/content and no dead padding. Each terminal clip
has18 transitions; an H16 window starting at2 includes its death at transition17.
Head validation pairs each of the18 VAL death windows with a low-HP, still-alive
window from the same episode. Those36 diagnostic windows contain576 measured
transitions but only324 unique transitions; they are not a population prevalence
estimate. A15-second data cutoff is not labeled player death.

The explicit Dreamer-style math uses REINFORCE for sampled discrete actions,
entropy3e-4, gamma332/333, lambda0.95,255-bin raw-space symexp two-hot values,
online stop-gradient bootstrap, a slow-value regularizer and0.98/0.02 target EMA.
Return scaling uses5th/95th percentile EMA at rate0.01 with scale floor1. The
compact game-specific actor/critic heads use512 units rather than claiming the
reference architecture. Serialized game-specific supports bound predicted reward
to **±2** and critic values to **±700**; raw-target clipping counters are reported.
Official defaults and the support adaptation are distinguished in the plan.
See the [official DreamerV3 implementation](https://github.com/danijar/dreamerv3)
and [reference configuration](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/configs.yaml).

Two discarded GPU probes actually completed: `probe_v2` ran four learner updates,
including three actor/critic updates at batch3; `probe_main18` repeated four
updates/three actor updates at the production batch18. Both preserved the frozen
world and passed gradient isolation checks. Their real Source policy validation
used **four starts and an8-transition horizon**, separately from the36-window
reward/continuation panel. At probe step4, death recall was **0**, with false-death
rate0: continuation remained underfit. These probes establish executable updates
and honest validation, not policy skill or a gameplay quality pass.
The MAIN18 probe also resumed its complete checkpoint and reached step8/seven
actor updates with fresh Source and head-validation reports and no training error.

The implementation checks passed **44 Python CPU tests in total**, including
the4 policy-validation/page-server tests, plus **6 JavaScript/DOM UI tests**.
The policy's production Source
validation is configured for **32 transitions every100 learner updates**; the
existing dynamics validation schedule is unchanged. The policy page is
[localhost:8787](http://127.0.0.1:8787/), with its server PID9100; the dynamics page
remains [localhost:8765](http://127.0.0.1:8765/). Local training artifacts are
ignored by Git and do not promise checkpoint availability in a clean clone.

[policy training plan](training_runs/dreamer_actor_critic_20261114/training_plan.json),
[production launch](training_runs/dreamer_actor_critic_20261114/launch.json),
[same-plan resume proof](training_runs/dreamer_actor_critic_20261114/resume_verification.json),
[production Source baseline100](training_runs/dreamer_actor_critic_20261114/validation/step_000100.json),
[production head validation100](training_runs/dreamer_actor_critic_20261114/validation/reward_step_000100.json),
[actual page capture128](training_runs/dreamer_actor_critic_20261114/diagnostics/ui_cpu/policy_8787.png),
[discarded world-refresh proof](training_runs/dreamer_actor_critic_20261114/diagnostics/refresh_probe.json),
[batch3 probe](training_runs/dreamer_actor_critic_probe_v2_20261114/status.json),
[MAIN18 probe](training_runs/dreamer_actor_critic_probe_main18_20261114/status.json),
[probe Source validation](training_runs/dreamer_actor_critic_probe_main18_20261114/validation/step_000004.json),
[probe death/alive head validation](training_runs/dreamer_actor_critic_probe_main18_20261114/validation/reward_step_000004.json).

## H112. Serial Source collection, motion only world and policy rounds

15 November2026, Moscow time; run directories retain their14 November start
names. The user requested motion-only dynamics and a real repeating cycle:
policy controls Source, RSSM learns from those new trajectories, then policy
learns in the updated RSSM. This explicitly supersedes the earlier standalone
policy stop only for the serial cycle; H111's stop at200 learner/100 actor
updates remains a correct historical event. Human game controls are unchanged.

The new protocols are `dreamer_motion_Source_world_policy_cycle_v1` for the
controller and `rssm_motion_only_main16_v1` for dynamics. MAIN16 is exactly
**8 no-enemy navigation +8 quiet NPC-motion**; Combat loss, combat MAIN/AUX/late,
combat RNG draws and independent contact draws are0. Player uses1/2 NAV+1/2
quiet; Enemy uses the quiet actors only. All three RSSM memories and their
existing Transformer/MoE architecture remain. Pure combat readouts are frozen;
shared combat memory may receive motion gradients. Short recovery uses actual
MAIN own-state/h/z prefixes rather than extra independent fragments.

The final user-approved ladder is **[16,48,80,112,144,180]**, **30 complete Adam
updates on each H**, **180 world updates per round**. Full BPTT spans current H,
reaching the **6-second /180-transition** target at30Hz. There is no TOTAL
threshold, streak or early advance. Ending H180/30 marks a completed world
phase, not a quality pass; the next outer round restarts H16 but preserves
cumulative weights/Adam/sampler history. Earlier H450/MAIN18 gates are archived
without reinterpreting their counters or old loss units.

The exact preserved world source is **6873**, SHA256
`531405a5721d6236fdedd2f5b42a667df5c12aef6ee46604935ca56f0251070c`.
The full pre-SGD conservation proof passed for **1599 model tensors, all442
named Adam states/seven groups, every moment/scalar, CPU RNG, all CUDA RNG states
(one device),
global PCG, raw private Source samplers and typed parent ledgers**. The new
revision owns its8+8 sampler clocks and stage/round counters; old MoE, player,
enemy-motion/contact and combat lineage remains opaque history. VAL pins the
original held-out corpora/panels for **180 transitions**; old450 reports remain
historical diagnostics. Future Source states/images never correct imagined or
free RSSM trajectories.

Collection actually completed **32 independent real Source TRAIN episodes**:
16 navigation +16 quiet enemy, each180 transitions, **5760 transitions** in
**48.606 seconds of collection wall time**. Firing was masked, exploration
ε=0.30, Source state repairs0; policy/world parameter hashes stayed exact.
The collector used pinned policy **200/100 actor updates** and world6873.
Each world8-actor cohort samples **4 original +4 newest Source episodes**;
earlier appended corpora are retained as provenance, not substituted for the
original half. VAL remains separate and unchanged.

After each180-update world round, policy explicitly refreshes to that complete
checkpoint, performs **20 head-only reward/continue recalibration updates**,
then **50 actual actor/critic updates**, with world and image assets frozen.
Policy MAIN18 remains6 NAV/6 quiet/6 combat; each NAV/quiet group includes
3 newest Source +3 original starts. Imagined horizon remains15 prior-only
transitions. The next collection uses the updated policy. Heads-only work is
excluded from the actor-update counter and does not change DYN16.

Preflight passed **13 checks**. Full disposable cycle smoke performed one real
world update **6873→6874** with Source appended, then explicitly rebound policy
and reached **222 learner/102 actor updates** (20 head updates +2 actor updates).
Those smoke checkpoints were not promoted. Actual capacity probes with Adam
measured **H16:8.34s /1.91GiB** and **H180:70.82s /4.51GiB** peak allocation.
Approximately **2 hours per world round** is only a rough scaling estimate,
excluding variable collection/policy/validation overhead and any time needed
to reach quality.

Production controller **PID22908** and dynamics child **PID52828** started.
At **6903**, production completed **exactly30 H16 updates** and switched to
**H48/stage0/30**; policy is waiting for this world round. The subsequent actual
live UI check at6908 showed H48/stage5/30 and35 new updates. It accepted the real
boundary row30 as trainedH16/currentH48, retained35 current-plan chart rows,
and displayed the6903/H48 RF3/SR preview and map without a stage mismatch.
The cycle root is
`training_runs/dreamer_motion_cycle_20261114`. Run/resume uses
`python -m turkeyworld.dreamer_motion_cycle`; an explicit
`--resume-after-user-stop` archives cycle/world/policy stop markers before
resuming. Create **`training_runs/dreamer_motion_cycle_20261114/stop.request`**
to stop the whole cycle safely. Ordinary restarts never clear a user's stop
or automatically restart stopped children. Original stop flags were archived
once in `retired_initial_flags` when this newly authorized cycle began.

The dynamics page remains [localhost:8765](http://127.0.0.1:8765/), with a serial
cycle card; policy remains [localhost:8787](http://127.0.0.1:8787/).
On actual smoke6874 JSON plus its actual metrics file, Node verified the new
identity, loss-row pins, MAIN8+8, stage1/30 and Combat OFF. Its TOTAL **0.2836225**
used the new formula and does not establish improvement over old combat TOTAL.
Cached preview6873 remained correctly labelled: **RF3/NPR32317, Base31709,
decoder4125, frozen SR3000**, five rows and the same current-H16 map drift.
All image workers remain stopped/completed. The paired runtime/default launcher
budget is now180;19 CPU tests cover that change. The historical launcher still
assumes Atlas SR, so these checks do not claim a newly launched modern-asset
game window. No game window was reopened. Dashboard verification passed
**259 Node tests**; the full-state proof helper passed4 CPU tests including15
negative mutation cases, and the validation-pin helper passed8 CPU tests.

[Controller launch](training_runs/dreamer_motion_cycle_20261114/controller_launch.json),
[cycle plan](training_runs/dreamer_motion_cycle_20261114/cycle_plan.json),
[13-check preflight](training_runs/dreamer_motion_cycle_20261114/preflight.json),
[Source receipt](training_runs/dreamer_motion_cycle_20261114/cycles/cycle_000000/source/collection_receipt.json),
[world source pin](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/motion_only_source.json),
[complete initial conservation](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/motion_only_full_boundary_verification.json),
[motion plan](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/motion_only_plan.json),
[fixed VAL plan](training_runs/rssm_moe_nav15_enemy25_cap20_20261113/motion_only_validation_plan.json),
[discarded world smoke](training_runs/motion_cycle_smoke_world_20261114/status.json),
[discarded policy smoke](training_runs/motion_cycle_smoke_policy_20261114/status.json).
These are local, mostly ignored training artifacts; budget completion and
improved imagined reward do not certify Source gameplay quality.

## Continuing this journal

Add each new H stage to both language versions, preserving earlier results.
Each entry should state the request, change, steps actually executed,
protocol/panel, metric value, gate, artifact path and weight availability.
Current `status.json` changes; numerical conclusions should be pinned to a
specific step and report. Record budget completion, user acceptance and quality
passes as separate events. The main instructions remain in the READMEs.

Most local `training_runs/` are excluded from Git: their links work in this
workspace and do not promise data/weight availability in a clean clone.
