"""Real non-unit RF normalization survives the format5 named-weight overlay."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import test_renderer_rf5_bundle as support
from test_renderer_enemy_objectives import batch
from turkeyworld.decoder_rf_adaptation_bundle import export_decoder_adaptation
from turkeyworld.renderer_rf5_bundle import export_rf5_renderer
from turkeyworld.renderer_rf_codec_bundle import export_rf_codec_renderer
from turkeyworld.renderer_student import load_renderer_student


class RFCodecNonunitTests(unittest.TestCase):
    def test_actual_nested_format5_keeps_nonzero_mean_nonunit_std_and_RGB_exact(self):
        torch.set_num_threads(1)
        original=support.NeuralWorldModel
        def nonunit(config):
            result=original(config)
            with torch.no_grad():
                result.flow_renderer.latent_mean.copy_(torch.linspace(-.12,.17,16).reshape(1,16,1,1))
                result.flow_renderer.latent_std.copy_(torch.linspace(.4,.9,16).reshape(1,16,1,1))
            return result
        class Fixture(support.RendererRF5BundleTests):pass
        with patch.object(support,'NeuralWorldModel',side_effect=nonunit):Fixture.setUpClass()
        self.addCleanup(Fixture.tearDownClass)
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);source=deepcopy(Fixture.unloaded)
            model=deepcopy(Fixture.expanded);model.flow_renderer.steps=3
            rf3=root/'rf3.pt';export_rf5_renderer(model,Fixture.parent_path,rf3,
                training_steps=29612,local_updates=3,source_model=source)
            current=deepcopy(source);load_renderer_student(current,rf3)
            with torch.no_grad():current.flow_renderer.codec.decoder.conv_out.bias.add_(.1)
            decoder=root/'decoder.pt';export_decoder_adaptation(current,rf3,decoder,
                decoder_updates=20,source_model=source)
            working=deepcopy(source);load_renderer_student(working,decoder)
            with torch.no_grad():working.flow_renderer.output.bias.add_(.03)
            path=root/'rf5format.pt';export_rf_codec_renderer(working,decoder,path,
                local_updates=3,source_model=source)
            loaded=deepcopy(source);load_renderer_student(loaded,path)
            self.assertTrue(torch.equal(working.flow_renderer.latent_mean,loaded.flow_renderer.latent_mean))
            self.assertTrue(torch.equal(working.flow_renderer.latent_std,loaded.flow_renderer.latent_std))
            self.assertGreater(float(loaded.flow_renderer.latent_mean.abs().sum()),0)
            self.assertFalse(torch.equal(loaded.flow_renderer.latent_std,torch.ones_like(loaded.flow_renderer.latent_std)))
            data=batch((False,));features=torch.zeros(1,128)
            for net in(working,loaded):net.eval().requires_grad_(False)
            with torch.inference_mode():
                a=working.render(data['state'],features,working.encode_map(data['map']))
                b=loaded.render(data['state'],features,loaded.encode_map(data['map']))
            self.assertTrue(torch.equal(a,b))


if __name__=='__main__':unittest.main()
