"""Validate a pinned control and a matched STE continuation. Never edits live weights."""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from .training_pipeline import atomic_json, now
from .process_runtime import process_alive, process_identity, process_identity_alive
from .vjepa_independent_panel import INDEPENDENT_FINGERPRINT

STREAMS = ('navigation', 'enemy_motion')


def read(path):
    return json.loads(path.read_text())


def compare(plan, branches, receipts):
    """Refuse a decision if actions/samples, update budgets or VAL panels differ."""
    start, end = plan['start_step'], plan['target_step']
    def sampling(rows):
        if [r['step'] for r in rows] != list(range(start+1, end+1)):
            raise ValueError('Incomplete update history')
        for r in rows:
            if r['horizon'] != 180 or r['main_actors'] != 16 or r['wall_coefficient'] != .25 or len(r['samples']) != 16:
                raise ValueError('Non-matched objective or batch size')
        return [r['samples'] for r in rows]

    if sampling(receipts['hard']) != sampling(receipts['ste']):
        raise ValueError('TRAIN windows/order differ between branches')
    info = {}
    for name, report in branches.items():
        if any(r.get('player_clamp_gradient', 'hard') != name for r in receipts[name]):
            raise ValueError('Unexpected TRAIN clamp mode')
        if (report['step'] != end or report['panel_fingerprint'] != INDEPENDENT_FINGERPRINT
                or report['panel'] != 'independent' or report['feedback_mode'] != 'on'
                or report['horizon'] != 180 or report['objective_wall_coefficient'] != .25
                or report['objective_auxiliary_mode'] != 'blocks16' or report['Source_future_feedback']):
            raise ValueError('Unexpected VAL condition')
        m = {}
        for stream in STREAMS:
            s = report[stream]
            if s['count'] != 8: raise ValueError('Partial validation')
            m[stream+'_player'] = s['player_XY_H180_mean_tiles']
            m[stream+'_wall_ticks'] = s['player_wall_overlap_ticks_mean']
        m['enemy_H180'] = report['enemy_motion']['enemy_XY_H180_mean_tiles']
        for key in ('false_stops','free_axes','false_passes','blocked_axes'):
            m[key] = sum(report[s]['common_start_axis'][key] for s in STREAMS)
        if not all(math.isfinite(v) for v in m.values()): raise ValueError('Nonfinite metric')
        info[name] = m
    a, b = info['hard'], info['ste']
    if any(a[k] != b[k] or a[k] == 0 for k in ('free_axes','blocked_axes')):
        raise ValueError('Source probe denominators differ or are empty')
    checks = {'fewer_false_stops': b['false_stops'] < a['false_stops'],
              'no_extra_false_passes': b['false_passes'] <= a['false_passes'],
              'no_worse_enemy': b['enemy_H180'] <= a['enemy_H180']}
    for s in STREAMS:
        checks[s+'_lower_player_drift'] = b[s+'_player'] < a[s+'_player']
        checks[s+'_no_more_wall_ticks'] = b[s+'_wall_ticks'] <= a[s+'_wall_ticks']
    return dict(status='complete', measurements=info, checks=checks, matched_train_windows=True,
                panel_fingerprint=INDEPENDENT_FINGERPRINT, passes_pilot=all(checks.values()),
                deployed=False, pilot_updates=end-start, completed_at=now())


def write_report(root, result):
    a, b = (result['measurements'][k] for k in ('hard', 'ste'))
    stats = [('Дрейф игрока H180: навигация, клетки','navigation_player'),
             ('Дрейф игрока H180: преследование, клетки','enemy_motion_player'),
             ('Дрейф врага H180, клетки','enemy_H180'),
             ('Тики пересечения стен: навигация','navigation_wall_ticks'),
             ('Тики пересечения стен: преследование','enemy_motion_wall_ticks')]
    lines = ['# Проверка градиента движения игрока', '',
             f"По {result['pilot_updates']} обновления от одного снимка, на идентичных TRAIN-окнах; LR 1e-5, wall 0,25. "
             'Панель H180: по восемь других эпизодов на поток. Ошибки команд измерены отдельно, из одинаковых исходных состояний.',
             '', '| Метрика | Обычный градиент | STE |', '|---|---:|---:|']
    for label, key in stats:
        lines.append(f'| {label} | {a[key]:.4f} | {b[key]:.4f} |')
    for label, num, den in [('Ложные остановки','false_stops','free_axes'),('Ложные проходы','false_passes','blocked_axes')]:
        lines.append(f'| {label} | {a[num]}/{a[den]} | {b[num]}/{b[den]} |')
    conclusion = ('Все условия короткого пилота выполнены.' if result['passes_pilot']
                  else 'Условия для перехода на STE не выполнены.')
    lines += ['', conclusion + ' Веса рабочего цикла не заменялись.', '']
    (root/'REPORT_RU.md').write_text('\n'.join(lines),encoding='utf8')


def score_reports(root, plan):
    reports, receipts = {}, {}
    for name in ('hard','ste'):
        reports[name] = read(root/name/'validation.json')
        receipts[name] = [json.loads(line) for line in (root/name/'metrics.jsonl').read_text().splitlines() if line]
    result = compare(plan, reports, receipts)
    write_report(root, result)
    atomic_json(root/'result.json',result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--cycle', type=Path, help='Active cycle whose stop request is respected')
    p.add_argument('--pinned-control-evaluated', action='store_true', help='Reuse separate completed validation of the pinned hard control in this experiment')
    p.add_argument('--reports-only', action='store_true', help='Score existing paired receipts without waiting or running GPU jobs')
    a = p.parse_args()
    root = a.experiment.resolve()
    plan = read(root/'plan.json')
    if a.reports_only:
        score_reports(root, plan)
        return
    if a.cycle is None: p.error('--cycle is required in monitor mode')
    cycle = a.cycle.resolve()
    def status(phase, **kw):
        atomic_json(root/'progress.json', dict(phase=phase, updated_at=now(), **kw))
    def stop_check():
        if (cycle/'stop.request').exists() or (root/'stop.request').exists():
            (root/'ste/stop_training').touch()
            raise InterruptedError('Stop requested')
    try:
        pid = int((root/'ste/train.pid').read_text())
        trainer_identity = process_identity(pid) if process_alive(pid) else None
        status('training')
        while trainer_identity is not None and process_identity_alive(trainer_identity):
            stop_check()
            time.sleep(2)
        stop_check()
        if read(root/'ste/status.json')['world_updates'] != plan['target_world_updates']:
            raise RuntimeError('Candidate did not finish; see ste/train.log')
        for name in ('hard', 'ste'):
            arm = root/name
            if name == 'hard' and a.pinned_control_evaluated:
                # score_reports rejects partial, mismatched, or differently configured panels.
                if not (arm/'validation.json').exists(): raise FileNotFoundError('Missing control validation')
                continue
            args = [sys.executable, '-m', 'turkeyworld.validate_vjepa_sr', '--world',
                    str(arm/'checkpoints/vjepa_world.pt'), '--panel','independent',
                    '--output',str(arm/'validation.json'),'--wall-weight','.25','--auxiliary-mode','blocks16']
            status('validation', branch=name)
            with (arm/'validation.log').open('a') as f:
                proc = subprocess.Popen(args, stdout=f, stderr=f)
                try:
                    while proc.poll() is None:
                        stop_check()
                        time.sleep(2)
                finally:
                    if proc.poll() is None: proc.terminate(); proc.wait()
                if proc.returncode: raise RuntimeError('Validator failed: '+name)
        result = score_reports(root, plan)
        status('complete',passes_pilot=result['passes_pilot'])
    except BaseException as exc:
        status('stopped' if isinstance(exc,InterruptedError) else 'failed',error=str(exc))
        raise


if __name__ == '__main__':
    main()
