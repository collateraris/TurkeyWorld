import unittest
import torch
from turkeyworld.models import NeuralWorldModel,WorldModelConfig
from turkeyworld.grow_vjepa_dynamics import grow
class GrowthTests(unittest.TestCase):
 def test_function_adam_ema_preserved_and_new_block_gradient(self):
  torch.set_num_threads(2)
  model=NeuralWorldModel(WorldModelConfig(kind='teacher',image_size=128,architecture='vjepa2_ac',endless=True,renderer_mode='flow_matching',flow_steps=1,flow_width=32,flow_depth=1,flow_heads=4,vjepa_width=32,vjepa_encoder_depth=1,vjepa_predictor_depth=1,vjepa_heads=4,vjepa_patch_size=16,vjepa_pooled_memory=True)).eval()
  names=[n for n,p in model.named_parameters()];opt=torch.optim.AdamW(model.parameters(),lr=1e-5)
  first=next(model.parameters());first.grad=torch.ones_like(first);opt.step();opt.zero_grad()
  saved=dict(optimizer_parameter_names=[names],optimizer=opt.state_dict(),ema_target_encoder={'encoder.'+k:v.detach().clone() for k,v in model.core.encoder.state_dict().items()},vjepa_sr_training={'updates':500,'frozen_parameters_sha256':'old'})
  image=torch.rand(1,4,3,128,128);actions=torch.randn(1,4,4);states=torch.randn(1,4,20)
  with torch.no_grad():old=model.core.encode_video(image);pred=model.core.predict(old,actions,states)
  model,payload,report=grow(model,saved,2,3)
  new=model.core.encode_video(image);new_pred=model.core.predict(new,actions,states)
  torch.testing.assert_close(new,old,rtol=0,atol=0);torch.testing.assert_close(new_pred,pred,rtol=0,atol=0)
  for i,entry in saved['optimizer']['state'].items():
   for key,value in entry.items():torch.testing.assert_close(payload['optimizer']['state'][i][key],value,rtol=0,atol=0)
  for key,value in saved['ema_target_encoder'].items():torch.testing.assert_close(payload['ema_target_encoder'][key],value,rtol=0,atol=0)
  new_pred.square().mean().backward()
  self.assertGreater(float(model.core.predictor.blocks[-1].attention.proj.weight.grad.abs().sum()),0)
  self.assertEqual(payload['vjepa_sr_training']['updates'],500)
  self.assertNotIn('frozen_parameters_sha256',payload['vjepa_sr_training'])
  self.assertGreater(report['new_parameter_tensors'],0)
