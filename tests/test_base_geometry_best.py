"""Best selection is rebuilt from exact raw-Base validation reports."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from test_train_base_geometry import PANEL, PLAN, WORLD, report, Loop, evaluator
from unittest.mock import patch
from turkeyworld.base_geometry_best import base_geometry_measurement, validate_saved_base_best


PINS=dict(panel_fingerprint=PANEL,plan_sha256=PLAN,feature_world_sha256=WORLD)


def measured(ssim, step):
    return dict(report(ssim,step),plan_sha256=PLAN,feature_world_sha256=WORLD)


def saved(root, values=((0,.74),(100,.75),(200,.75)), *, step=250):
    directory=Path(root)/'validation';directory.mkdir(exist_ok=True)
    history,best=[],None
    for tick,score in values:
        item=measured(score,tick);payload=json.dumps(item).encode()
        (directory/f'step_{tick:06d}.json').write_bytes(payload)
        sha=hashlib.sha256(payload).hexdigest()
        history.append(dict(step=tick,ssim=score,report_sha256=sha))
        if best is None or score>best['ssim']:
            best=dict(step=tick,ssim=score,metric='models.base_new.summary.ssim_mean',**PINS,report_sha256=sha)
    return dict(step=step,validation_history=history,best=best)


class BaseGeometryBestTests(unittest.TestCase):
    def test_actual_driver_saved_reports_roundtrip_earliest_best_without_latest_report_dependency(self):
        with tempfile.TemporaryDirectory() as root:
            trainer=Loop(root,start=0,scores={0:.75,100:.74,200:.75,300:.751})
            trainer.prepare()
            with patch('turkeyworld.train_base_geometry.evaluate_base_geometry',side_effect=evaluator(trainer)):
                for tick in (0,100,200):trainer.step=tick;trainer.validate(None)
                data=trainer.payload()
                # A partial/latest view is not the authoritative history.
                (Path(root)/'validation.json').write_text('{}',encoding='utf-8')
                best,history=validate_saved_base_best(data,run_dir=root,**PINS)
                self.assertEqual(best['step'],0);self.assertEqual(len(history),3)
                self.assertEqual(data['best'],best)
                trainer.step=300;trainer.validate(None)
                best,_=validate_saved_base_best(trainer.payload(),run_dir=root,**PINS)
                self.assertEqual(best['step'],300)

    def test_NaN_future_nonmaximum_and_latest_tie_best_cannot_block_new_best(self):
        with tempfile.TemporaryDirectory() as root:
            good=saved(root)
            for change in (dict(ssim=float('nan')),dict(ssim=.99),dict(step=999),
                           dict(step=200),dict(plan_sha256='0'*64),dict(feature_world_sha256='0'*64)):
                bad=deepcopy(good);bad['best'].update(change)
                with self.subTest(change=change),self.assertRaises(ValueError):
                    validate_saved_base_best(bad,run_dir=root,**PINS)
            best,history=validate_saved_base_best(good,run_dir=root,**PINS)
            best['ssim']=0;history.clear()
            self.assertEqual(good['best']['ssim'],.75)
            self.assertEqual(len(good['validation_history']),3)

    def test_tampered_missing_or_rehashed_changed_measurement_reports_reject(self):
        with tempfile.TemporaryDirectory() as root:
            good=saved(root)
            path=Path(root)/'validation/step_000100.json'
            original=path.read_bytes();path.write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError,'SHA'):
                validate_saved_base_best(good,run_dir=root,**PINS)
            changed=measured(.99,100);payload=json.dumps(changed).encode();path.write_bytes(payload)
            good['validation_history'][1]['report_sha256']=hashlib.sha256(payload).hexdigest()
            with self.assertRaisesRegex(ValueError,'disagrees'):
                validate_saved_base_best(good,run_dir=root,**PINS)
            path.unlink()
            with self.assertRaisesRegex(ValueError,'missing'):
                validate_saved_base_best(good,run_dir=root,**PINS)

    def test_rehashed_wrong_plan_or_changed_frozen_RF_and_codec_domain_reject(self):
        for change in (dict(panel_fingerprint='0'*64),dict(plan_sha256='0'*64),
                       dict(feature_world_sha256='0'*64),dict(frozen_flow_sha256='0'*64),
                       dict(codec_sha256='0'*64),dict(data_identity_sha256='0'*64)):
            with tempfile.TemporaryDirectory() as root:
                good=saved(root);item=measured(.75,100);item.update(change)
                payload=json.dumps(item).encode()
                (Path(root)/'validation/step_000100.json').write_bytes(payload)
                good['validation_history'][1]['report_sha256']=hashlib.sha256(payload).hexdigest()
                with self.subTest(change=change),self.assertRaises(ValueError):
                    validate_saved_base_best(good,run_dir=root,**PINS)

    def test_only_full_native_Base_without_RF_or_restorer_can_supply_measurement(self):
        for name,value in (('frames',4),('frames',256.),('native_resolution',128),
                ('kind','conditional_restorer_validation_v1'),('protocol','old'),
                ('source_held_out',False),('RF_forward_calls',1),('RF_forward_calls',False),
                ('restorer_active',True),('Source_Q_image_branch',True),('world_rollout',True),
                ('Source_RGB_into_Base_conditioning',True)):
            bad=measured(.75,100);bad[name]=value
            with self.subTest(name=name),self.assertRaises(ValueError):
                base_geometry_measurement(bad,**PINS)
        for score in (True,float('nan'),float('inf'),85.,1.01):
            with self.assertRaises(ValueError):base_geometry_measurement(measured(score,0),**PINS)
        good=measured(.74,0)
        good['models']['base_codec_new']=dict(summary=dict(frames=256,ssim_mean=1.))
        self.assertEqual(base_geometry_measurement(good,**PINS)['ssim'],.74)

    def test_initial_zero_only_and_history_must_start_zero_with_ordered_nonfuture_clocks(self):
        initial=dict(step=0,best=None,validation_history=[])
        self.assertEqual(validate_saved_base_best(initial,run_dir=None,**PINS),(None,[]))
        for change in (dict(step=1),dict(best={}),dict(validation_history=None)):
            with self.assertRaises(ValueError):
                validate_saved_base_best(dict(initial,**change),run_dir=None,**PINS)
        with tempfile.TemporaryDirectory() as root:
            good=saved(root)
            for change in ('missing0','reversed','future'):
                bad=deepcopy(good)
                if change=='missing0':bad['validation_history'].pop(0)
                if change=='reversed':bad['validation_history'].reverse()
                if change=='future':bad['step']=199
                with self.subTest(change=change),self.assertRaises(ValueError):
                    validate_saved_base_best(bad,run_dir=root,**PINS)


if __name__=='__main__':unittest.main()
