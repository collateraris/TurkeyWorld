"""Archive and resume one stopped run with a new inference-only preview revision.

Prepare/check do not deserialize checkpoints or change training state. Start
uses the old complete command with exactly one replacement: the explicit
preview revision path. The trainer owns training.pid and all checkpoint/state
restoration. This operator does not create a curriculum or reset any counters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from .training_pipeline import atomic_json, now
from .vqgan_pipeline import process_alive


REVISION_FLAG = '--preview-assets-revision-plan'


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _flag_value(argv, flag):
    positions = [i for i, value in enumerate(argv) if value == flag]
    if len(positions) != 1 or positions[0]+1 >= len(argv) or argv[positions[0]+1].startswith('--'):
        raise ValueError('Expected exactly one explicit argument '+flag)
    return argv[positions[0]+1]


def preview_revision_argv(original, revision_plan, run_dir):
    """Keep executable, source checkpoint, all plans and hyperparameters exact."""
    if (not isinstance(original, list) or not all(isinstance(value, str) for value in original)
            or '-m' not in original or original[original.index('-m')+1] != 'turkeyworld.train_rssm_combat'):
        raise ValueError('Expected the original structured RSSM combat training argv')
    if Path(_flag_value(original, '--run-dir')).resolve() != Path(run_dir).resolve():
        raise ValueError('Preview-only continuation must retain the exact same run directory')
    _flag_value(original, '--source-checkpoint')
    _flag_value(original, '--preview-assets-plan')
    _flag_value(original, REVISION_FLAG)
    _flag_value(original, '--loss-gated-curriculum-plan')
    result = list(original)
    result[result.index(REVISION_FLAG)+1] = str(Path(revision_plan).resolve())
    return result


def _stopped(root):
    state = _json(root/'status.json')
    launch = _json(root/'launch_command.json')
    if state.get('status') != 'stopped':
        raise ValueError('Root must stop the worker at a complete saved optimizer boundary first')
    pids = [launch.get('pid')]
    if (root/'training.pid').exists():
        pids.append(int((root/'training.pid').read_text(encoding='utf-8').strip()))
    if any(pid and process_alive(int(pid)) for pid in pids):
        raise ValueError('The old RSSM worker is still alive; never start a second worker')
    step = state.get('global_training_step')
    gate = state.get('rssm_training', {}).get('loss_gated_curriculum', {})
    if (type(step) is not int or step < 1 or gate.get('last_optimizer_step') != step
            or gate.get('settings', {}).get('required_consecutive_passes') != 10):
        raise ValueError('Need the preserved complete ten-consecutive-loss curriculum boundary')
    return state, launch, step


def _resolve(initial_path, revision_plan, saved_revision):
    from .rssm_preview_assets import resolve_preview_assets_plan
    raw = _json(revision_plan)
    if raw.get('version') != 2 or raw.get('kind') != 'preview_assets_revision_v2':
        raise ValueError('A further preview update needs the immutable chained v2 revision')
    initial, selected, record = resolve_preview_assets_plan(initial_path, revision_plan, saved_revision)
    if (record is None or record == saved_revision or record.get('preview_only') is not True
            or record.get('training_assets_changed') is not False or record.get('game_assets_changed') is not False):
        raise ValueError('The new revision must change only inference preview assets')
    return initial, selected, record


def _pins(argv, root):
    result = {}
    for value in argv:
        if value.startswith('--') and (value.endswith('-plan') or value in {'--source-checkpoint', '--renderer-checkpoint', '--sr'}):
            path = Path(_flag_value(argv, value)).resolve()
            result[str(path)] = _sha(path)
    base = root/'combat_training_plan.json'
    if not base.is_file():
        raise FileNotFoundError('Missing original immutable combat training plan')
    result[str(base)] = _sha(base)
    return result


def _archive(source, destination, *, hardlink=False):
    if destination.exists():
        raise FileExistsError('Never overwrite an existing immutable preview-resume archive')
    if hardlink:
        try:
            os.link(source, destination)
            return 'hardlink'
        except OSError:
            pass
    shutil.copy2(source, destination)
    return 'copy'


def prepare_preview_revision(run_dir, revision_plan, *, expected_step=None, expected_checkpoint_sha256=None):
    root, revision_plan = Path(run_dir).resolve(), Path(revision_plan).resolve()
    state, launch, step = _stopped(root)
    if expected_step is not None and expected_step != step:
        raise ValueError('Stopped optimizer step differs from the explicitly requested boundary')
    argv = preview_revision_argv(launch['argv'], revision_plan, root)
    old_revision = state.get('preview_assets_revision')
    if not isinstance(old_revision, dict):
        raise ValueError('Need the existing saved preview revision; never clear it to bypass lineage')
    old_path = Path(_flag_value(launch['argv'], REVISION_FLAG)).resolve()
    if (str(old_path) != old_revision.get('revision_plan_path')
            or _sha(old_path) != old_revision.get('revision_sha256')):
        raise ValueError('Saved preview revision and old launch disagree')
    initial, selected, record = _resolve(_flag_value(argv, '--preview-assets-plan'), revision_plan, old_revision)
    checkpoint = root/'checkpoints/rssm_world.pt'
    source_sha = _sha(checkpoint)
    if expected_checkpoint_sha256 is not None and source_sha != expected_checkpoint_sha256:
        raise ValueError('Stopped complete checkpoint SHA256 differs from the pinned source')
    pins = _pins(launch['argv'], root)
    pins.update(_pins(argv, root))
    for plan in (initial, selected):
        pins[plan['plan_path']] = plan['plan_sha256']
        for asset in ('renderer', 'sr'):
            pins[plan[asset]['checkpoint']] = plan[asset]['file_sha256']
    archive = root/'preview_revision_resumes'/f'g{step}_{record["revision_sha256"][:12]}'
    if archive.exists():
        raise FileExistsError('This boundary/revision archive already exists; use --check-prepared')
    archive.mkdir(parents=True)
    method = _archive(checkpoint, archive/'checkpoint_before.pt', hardlink=True)
    _archive(root/'launch_command.json', archive/'launch_before.json')
    _archive(root/'status.json', archive/'status_before.json')
    if source_sha != _sha(checkpoint) or source_sha != _sha(archive/'checkpoint_before.pt'):
        raise ValueError('Checkpoint changed while archiving the stopped complete boundary')
    receipt = {
        'version': 1, 'kind': 'same_run_preview_revision_resume_v1', 'status': 'prepared',
        'prepared_at': now(), 'run_dir': str(root), 'source_step': step,
        'source_checkpoint': str(checkpoint), 'source_checkpoint_sha256': source_sha,
        'checkpoint_backup': str(archive/'checkpoint_before.pt'), 'backup_method': method,
        'checkpoint_step_deserialized_by_operator': False,
        'checkpoint_step_verified_by_trainer_restore_required': True,
        'old_launch_archive': str(archive/'launch_before.json'), 'old_launch_sha256': _sha(root/'launch_command.json'),
        'old_status_archive': str(archive/'status_before.json'), 'old_status_sha256': _sha(root/'status.json'),
        'old_argv': list(launch['argv']), 'argv': argv,
        'only_argv_change': REVISION_FLAG, 'original_preview_revision': old_revision,
        'new_preview_revision': record, 'revision_plan': str(revision_plan),
        'revision_plan_sha256': _sha(revision_plan), 'immutable_input_pins': pins,
        'loss_gated_curriculum_at_boundary': state['rssm_training']['loss_gated_curriculum'],
        'model_Adam_RNG_sampler_EMA_and_curriculum_not_mutated_by_operator': True,
        'training_pid_file_owned_by_trainer': True, 'game_window_untouched': True,
        'required_restore_proof': str(root/f'resume_verification_g{step}_loss_gated_drift_v1.json'),
        'post_preview_model_Adam_RNG_verification_required': True,
    }
    atomic_json(archive/'resume.json', receipt)
    return archive/'resume.json', receipt


def check_prepared_preview_revision(run_dir, revision_plan, receipt_path):
    root, revision_plan = Path(run_dir).resolve(), Path(revision_plan).resolve()
    receipt = _json(receipt_path)
    state, launch, step = _stopped(root)
    if (receipt.get('kind') != 'same_run_preview_revision_resume_v1' or receipt.get('run_dir') != str(root)
            or receipt.get('revision_plan') != str(revision_plan) or receipt.get('source_step') != step
            or receipt.get('old_argv') != launch.get('argv')
            or receipt.get('argv') != preview_revision_argv(launch['argv'], revision_plan, root)
            or receipt.get('original_preview_revision') != state.get('preview_assets_revision')
            or receipt.get('loss_gated_curriculum_at_boundary') != state['rssm_training']['loss_gated_curriculum']):
        raise ValueError('Prepared preview-only command, source boundary, history or gate changed')
    for path, expected in ((root/'checkpoints/rssm_world.pt', receipt['source_checkpoint_sha256']),
                           (Path(receipt['checkpoint_backup']), receipt['source_checkpoint_sha256']),
                           (root/'launch_command.json', receipt['old_launch_sha256']),
                           (Path(receipt['old_launch_archive']), receipt['old_launch_sha256']),
                           (root/'status.json', receipt['old_status_sha256']),
                           (Path(receipt['old_status_archive']), receipt['old_status_sha256']),
                           (revision_plan, receipt['revision_plan_sha256'])):
        if _sha(path) != expected:
            raise ValueError('Prepared immutable preview-resume checkpoint/metadata changed: '+str(path))
    for path, expected in receipt['immutable_input_pins'].items():
        if _sha(path) != expected:
            raise ValueError('An inherited immutable training/preview asset changed: '+path)
    _, _, record = _resolve(_flag_value(receipt['argv'], '--preview-assets-plan'), revision_plan,
                            receipt['original_preview_revision'])
    if record != receipt['new_preview_revision']:
        raise ValueError('Prepared preview chain changed')
    return receipt


def start_prepared_preview_revision(run_dir, revision_plan, receipt_path):
    root = Path(run_dir).resolve()
    receipt = check_prepared_preview_revision(root, revision_plan, receipt_path)
    start_path = Path(receipt_path).resolve().parent/'start.json'
    if start_path.exists():
        raise FileExistsError('This prepared resume was already launched; never spawn a duplicate worker')
    with (root/'training.stdout.log').open('ab') as out, (root/'training.stderr.log').open('ab') as err:
        child = subprocess.Popen(receipt['argv'], cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.DEVNULL, stdout=out, stderr=err, close_fds=True,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    started = {'pid': child.pid, 'started_at': now(), 'source_step': receipt['source_step'],
               'receipt': str(Path(receipt_path).resolve()), 'receipt_sha256': _sha(receipt_path),
               'revision_plan': receipt['revision_plan'], 'revision_plan_sha256': receipt['revision_plan_sha256'],
               'checkpoint_and_curriculum_reset': False, 'training_pid_file_not_written_by_operator': True}
    atomic_json(start_path, started)
    launch = _json(receipt['old_launch_archive'])
    launch.update(argv=receipt['argv'], previous_pid=launch.get('pid'), pid=child.pid,
                  resumed_at=started['started_at'], resume_reason='user_requested_latest_renderer_preview_only',
                  preview_assets_revision_plan=receipt['revision_plan'],
                  preview_revision_resume_receipt=str(Path(receipt_path).resolve()),
                  preview_revision_source_step=receipt['source_step'], preview_only_revision=True,
                  model_Adam_RNG_and_curriculum_preservation_required=True)
    atomic_json(root/'launch_command.json', launch)
    return started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--revision-plan', type=Path, required=True)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--expected-step', type=int)
    parser.add_argument('--expected-checkpoint-sha256')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--check-prepared', action='store_true')
    mode.add_argument('--start-prepared', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        path, receipt = prepare_preview_revision(args.run_dir, args.revision_plan,
            expected_step=args.expected_step, expected_checkpoint_sha256=args.expected_checkpoint_sha256)
        print(json.dumps({'status': 'prepared', 'source_step': receipt['source_step'], 'receipt': str(path)}))
    else:
        if args.receipt is None:
            parser.error('Check/start requires the exact prepared --receipt')
        result = (start_prepared_preview_revision(args.run_dir, args.revision_plan, args.receipt)
                  if args.start_prepared else check_prepared_preview_revision(args.run_dir, args.revision_plan, args.receipt))
        print(json.dumps({'status': 'started' if args.start_prepared else 'checked',
                          'source_step': result['source_step'], 'pid': result.get('pid')}))


if __name__ == '__main__':
    main()
