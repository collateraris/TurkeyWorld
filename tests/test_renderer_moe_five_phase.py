from copy import deepcopy
import random
import unittest

import numpy as np
import torch

from test_renderer_moe_phase import Model, Stream
from turkeyworld.renderer_moe import RendererTokenMoE, _append_five_experts
from turkeyworld.renderer_moe_phase import renderer_moe_protocol
from turkeyworld.renderer_moe_five_phase import (renderer_moe_five_protocol, identity_digest,
    build_renderer_moe_five_optimizer, restore_renderer_moe_five_parent,
    renderer_five_parameter_groups, original_renderer_configuration, validate_renderer_moe_five_resume)
from turkeyworld.renderer_native_phase import _same
from turkeyworld.renderer_refinement import frozen_fingerprint


def fixture():
    model = Model()
    for head in (model.base_student, model.flow_renderer):
        head.output_moe = RendererTokenMoE(3, 3, 8, 2)
    old_groups = [list(model.base_student.body.parameters()), list(model.flow_renderer.body.parameters()),
        list(model.flow_renderer.geometry_opacity.parameters()),
        list(model.base_student.output_moe.parameters()) + list(model.flow_renderer.output_moe.parameters())]
    optimizer = torch.optim.AdamW([dict(params=g, lr=lr) for g, lr in
        zip(old_groups, (.0001, .0001, .00005, .0001))], weight_decay=.01)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        sum(p.square().mean() for group in old_groups for p in group).backward(); optimizer.step()
    # Both split routers must start from zero, while pretrained old output/router are nonzero.
    for index in optimizer.state_dict()['param_groups'][2]['params']:
        optimizer.state_dict()['state'][index]['step'].fill_(1)
    model.flow_renderer.geometry_opacity.requires_grad_(False)
    frozen = frozen_fingerprint(model)
    identity = dict(world_sha256='a' * 64, feature_world_sha256='a' * 64, feature_world_step=5906,
        codec_sha256='c' * 64, train={'fingerprint': 'train'}, val={'fingerprint': 'val'},
        panel=[1, 2], panel_fingerprint='panel', geometry_wall_panel=[3, 4],
        renderer_moe_protocol=renderer_moe_protocol())
    saved = dict(base=deepcopy(model.base_student.state_dict()), flow=deepcopy(model.flow_renderer.state_dict()),
        optimizer=deepcopy(optimizer.state_dict()), step=2000, renderer_training_steps=28200,
        base_training_steps=27800, frozen_fingerprint=frozen, identity=deepcopy(identity),
        configuration={'steps': 2000, 'inherited_parent_configuration':
                       {'combat_data_root': 'combat', 'appearance_data_root': 'appearance', 'edge_weight': .1}},
        sampler={'draws': 224000, 'rng': np.random.default_rng(17).bit_generator.state},
        far_wall_aux_sampler={'draws': 48000}, geometry_wall_aux_sampler={'draws': 32000},
        rng_torch=torch.get_rng_state().clone(), rng_numpy=deepcopy(np.random.get_state()),
        rng_python=deepcopy(random.getstate()), rng_cuda=[])
    for head in (model.base_student, model.flow_renderer): _append_five_experts(head.output_moe)
    new_identity = {**deepcopy(identity), 'parent_identity_sha256': identity_digest(identity),
                    'renderer_moe_protocol': renderer_moe_five_protocol()}
    configuration = {'steps': 2000, 'inherited_parent_configuration': deepcopy(saved['configuration'])}
    return model, saved, new_identity, configuration, [Stream({}) for _ in range(3)], frozen


class RendererMoEFivePhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_all_four_old_groups_mixed_ages_options_rng_and_three_pcg_adopt_exactly(self):
        model, saved, identity, cfg, streams, frozen = fixture()
        optimizer = build_renderer_moe_five_optimizer(model, saved)
        before = deepcopy(model.state_dict())
        proof = restore_renderer_moe_five_parent(saved, model, optimizer, *streams, identity, cfg, frozen,
                                                 parent_checkpoint_sha256='d' * 64)
        self.assertTrue(_same(before, model.state_dict()))
        self.assertTrue(_same(saved['optimizer']['state'], optimizer.state_dict()['state']))
        self.assertEqual(saved['optimizer']['param_groups'], optimizer.state_dict()['param_groups'][:4])
        self.assertEqual(proof['parent_renderer_training_steps'], 28200)
        self.assertEqual(proof['parent_base_training_steps'], 27800)
        self.assertEqual(proof['start_step'], 0); self.assertFalse(proof['feature_world_changed_explicitly'])
        self.assertEqual(proof['added_parameter_tensors'], 52)
        self.assertEqual(proof['new_Adam_states_at_start'], 0)
        self.assertTrue(torch.equal(saved['rng_torch'], torch.get_rng_state()))
        self.assertTrue(_same(saved['rng_numpy'], np.random.get_state()))
        self.assertEqual(saved['rng_python'], random.getstate())
        for stream, key in zip(streams, ('sampler', 'far_wall_aux_sampler', 'geometry_wall_aux_sampler')):
            self.assertTrue(_same(stream.sampler_state(), saved[key]))
        self.assertEqual(original_renderer_configuration(saved['configuration'])['combat_data_root'], 'combat')

    def test_old_and_new_experts_update_while_private_old_moments_age_and_frozen_assets_stay_exact(self):
        model, saved, identity, cfg, streams, frozen = fixture()
        optimizer = build_renderer_moe_five_optimizer(model, saved)
        restore_renderer_moe_five_parent(saved, model, optimizer, *streams, identity, cfg, frozen,
                                         parent_checkpoint_sha256='d' * 64)
        old = deepcopy(model.state_dict()); moments = deepcopy(optimizer.state_dict())
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            # Use every actual expert/router tensor; check Adam groups rather than mirroring the migration.
            sum(p.square().mean() + .01 * p.mean() for p in model.parameters() if p.requires_grad).backward()
            optimizer.step()
        self.assertEqual(frozen, frozen_fingerprint(model))
        for index in moments['param_groups'][2]['params']:
            self.assertTrue(_same(moments['state'][index], optimizer.state_dict()['state'][index]))
        for index in moments['param_groups'][3]['params']:
            self.assertEqual(float(optimizer.state_dict()['state'][index]['step']), 6.)
        for index in optimizer.state_dict()['param_groups'][4]['params']:
            self.assertEqual(float(optimizer.state_dict()['state'][index]['step']), 3.)
        self.assertTrue(any(not torch.equal(value, model.state_dict()[name])
                            for name, value in old.items() if '.experts.3.' in name))

    def test_tampered_parent_inputs_clone_split_options_and_missing_moments_rejected_before_adoption(self):
        for mode in ('world', 'panel', 'clone', 'split', 'option', 'missing_moment', 'old_weight', 'names'):
            model, saved, identity, cfg, streams, frozen = fixture()
            optimizer = build_renderer_moe_five_optimizer(model, saved)
            if mode == 'world': identity['feature_world_sha256'] = 'b' * 64
            elif mode == 'panel': identity['panel'] = [4, 5]
            elif mode == 'clone':
                with torch.no_grad(): model.base_student.output_moe.experts[3].linear1.weight.add_(.1)
            elif mode == 'split':
                with torch.no_grad(): model.base_student.output_moe.split_router.bias.add_(.1)
            elif mode == 'option': optimizer.param_groups[0]['lr'] = .02
            elif mode == 'missing_moment': saved['optimizer']['state'].pop(next(iter(saved['optimizer']['state'])))
            elif mode == 'old_weight':
                with torch.no_grad(): model.base_student.body.weight.add_(.1)
            else: saved['optimizer_parameter_names'] = [['wrong']] * 4
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                restore_renderer_moe_five_parent(saved, model, optimizer, *streams, identity, cfg, frozen,
                                                 parent_checkpoint_sha256='d' * 64)
            self.assertTrue(all(not stream.sampler_state() for stream in streams))

    def test_resume_retains_cumulative_clocks_and_pins_phase_names_and_full_parent(self):
        model, saved, identity, cfg, streams, frozen = fixture()
        optimizer = build_renderer_moe_five_optimizer(model, saved)
        phase = restore_renderer_moe_five_parent(saved, model, optimizer, *streams, identity, cfg, frozen,
                                                 parent_checkpoint_sha256='d' * 64)
        phase['configuration'] = cfg
        resume = dict(step=0, parent_step=2000, renderer_training_steps=28200, base_training_steps=27800,
            configuration=deepcopy(cfg), identity=deepcopy(identity), frozen_fingerprint=frozen,
            renderer_moe_phase=deepcopy(phase), renderer_moe_plan_sha256='e' * 64,
            optimizer_parameter_names=renderer_five_parameter_groups(model)[1], optimizer=optimizer.state_dict())
        def check(value):
            validate_renderer_moe_five_resume(value, phase, identity=identity, configuration=cfg, frozen=frozen,
                plan_sha256='e' * 64, parent_checkpoint_sha256='d' * 64)
        check(resume)
        later = deepcopy(resume); later.update(step=13, renderer_training_steps=28213, base_training_steps=27813)
        check(later)
        for key in ('renderer_training_steps', 'base_training_steps', 'parent_step', 'renderer_moe_plan_sha256',
                    'optimizer_parameter_names', 'renderer_moe_phase'):
            broken = deepcopy(later); broken[key] = 0
            with self.subTest(key=key), self.assertRaises(ValueError): check(broken)


if __name__ == '__main__': unittest.main()
