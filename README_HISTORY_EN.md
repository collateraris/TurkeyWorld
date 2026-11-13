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
