"""Explicit next-round schedule; never reinterpret an active or archived gate."""
from copy import deepcopy
import hashlib
import json

SETTINGS=dict(protocol='motion_round_48_112_180_40_v1',horizons=[48,112,180],
    updates_per_stage=40,world_updates_per_round=120)

H180_SETTINGS=dict(protocol='motion_round_180_100_v2',horizons=[180],updates_per_stage=100,world_updates_per_round=100)

def validate_plan(plan):
    if (plan.get('settings') not in (SETTINGS,H180_SETTINGS) or type(plan.get('effective_round'))is not int
            or plan['effective_round']<1):raise ValueError('Invalid next-round motion schedule')
    previous=plan.get('previous_revision')
    if previous is not None:
        validate_plan(previous['plan'])
        if plan['effective_round']<=previous['plan']['effective_round']:raise ValueError('Schedule extension must affect a later round')
    return plan


def gate_schedule(gate):
    schedule=gate.get('schedule')
    if schedule is None:return (16,48,80,112,144,180),30,180
    if schedule not in (SETTINGS,H180_SETTINGS):raise ValueError('Motion round schedule changed')
    return tuple(schedule['horizons']),schedule['updates_per_stage'],schedule['world_updates_per_round']


def schedule_for_round(revision,index):
    if revision is None:return None
    plan=validate_plan(revision['plan'])
    return deepcopy(plan['settings']) if index>=plan['effective_round'] else schedule_for_round(plan.get('previous_revision'),index)


def read_revision(path,previous=None):
    if not path.exists():
        if previous is not None:raise ValueError('Pinned next-round schedule is missing')
        return None
    raw=path.read_bytes();plan=validate_plan(json.loads(raw))
    revision=dict(plan=plan,plan_sha256=hashlib.sha256(raw).hexdigest())
    if previous is not None and revision!=previous and plan.get('previous_revision')!=previous:raise ValueError('Pinned next-round schedule changed')
    return revision
