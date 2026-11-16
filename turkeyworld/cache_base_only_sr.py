"""Cache direct Base32 with untouched Source128 labels; no RF or codec execution."""
import argparse,json
from pathlib import Path
import torch
from .models import load_world_checkpoint
from .renderer_student import load_renderer_student
from .base_only_renderer import enable_base_only
from .cache_sr_neural_pairs import build_cache
from .cache_sr_moe_pairs import render_profiled_cache_batch,ENVIRONMENT_POLICY
from .small_enemy_data import digest


def main():
    p=argparse.ArgumentParser()
    for key in ('world','renderer','source-root','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(2)
    model,saved=load_world_checkpoint(a.world,'cpu');step=saved['training_steps'];del saved
    load_renderer_student(model,a.renderer,allow_unqualified_codec=True)
    metadata=enable_base_only(model);model.to(a.device).float().eval().requires_grad_(False)
    identity=dict(renderer_input='base_only',base_only=metadata,world_sha256=digest(a.world),world_step=step,
        renderer_sha256=digest(a.renderer),future_rgb_features=False,paired_same_pose_map_tick=True,
        source_environment_policy=ENVIRONMENT_POLICY,flow_matching=False,codec=False)
    report=build_cache(a.source_root,a.output,model,cache_identity=identity,include_codec=False,
        render_batch_fn=render_profiled_cache_batch,batch_size=16)
    print(json.dumps({'status':report['status'],'frames':report['current']}),flush=True)

if __name__=='__main__':main()
