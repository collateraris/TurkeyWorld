TurkeyWorld v0.3 — technical verification, 2026-11-08

The JSON reports separate execution, state integrity, activity and sampled RGB.
No trained gameplay weights are included. No RTX 4090 benchmark was performed.

1. tbptt_training_9000_summary.json: one real continuous 9000-step supervised
   segment, 282 chunk optimizer updates, 281 gradient detach boundaries, exact
   numerical state/memory carry across boundaries. State-only; no RGB quality
   or gameplay correctness claim. Training checkpoint is not bundled.
2. event_v1_untrained_idle_300s.json: actual untrained neural model completed
   9000 finite state-only steps. Integrity passed; active-session gates failed.
   This is intentionally NOT a successful five-minute gameplay result.
3. reference_patrol_checker_9000.json: the reference environment's full-speed
   9000-step patrol passes the same read-only checker without false failures.
4. untrained_rgb_diagnostic.json: sampled-render interface probe. The untrained
   RGB-quality gate fails as expected.
5. Legacy CPU random-weight benchmark files are historical v0.1/v0.2 data,
   not current trained v0.3 or GPU performance.

Behavioral tests cover combat accounting, all 99 rounds, no health drift,
FP32 persistent state, camera/visibility contracts, rendering invariance,
TBPTT continuity, compatibility and evaluator fault injection.

Final suite: 51 tests passed on CPU. Correction collector actual CLI saved two independent aligned oracle branches without modifying neural state or memory.
