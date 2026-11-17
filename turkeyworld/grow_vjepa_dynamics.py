"""Function-preserving depth growth with named Adam and EMA continuation."""
import argparse,copy,json
from dataclasses import replace
from pathlib import Path
import torch
from .models import load_world_checkpoint
from .vjepa2_ac import VideoBlock,_identity_initialize
from .distill_renderer import _save
from .training_pipeline import atomic_json,now

def grow(model,saved,encoder_depth=34,predictor_depth=59):
    old_weights={k:v.detach().clone() for k,v in model.state_dict().items()}
    old_depths={name:len(getattr(model.core,name).blocks) for name in ('encoder','predictor')}
    for name,depth in [('encoder',encoder_depth),('predictor',predictor_depth)]:
        part=getattr(model.core,name)
        if depth<old_depths[name]:raise ValueError('Cannot discard trained blocks')
        for _ in range(depth-old_depths[name]):
            block=VideoBlock(model.core.width,model.config.vjepa_heads,model.config.vjepa_mlp_ratio)
            _identity_initialize(block);part.blocks.append(block)
    model.config=replace(model.config,vjepa_encoder_depth=encoder_depth,vjepa_predictor_depth=predictor_depth,vjepa_gradient_checkpointing=True)
    model.vjepa_backend.config=model.config
    for name,value in old_weights.items():
        if not torch.equal(value,model.state_dict()[name]):raise AssertionError('Old tensor changed: '+name)
    old_names=saved['optimizer_parameter_names']
    if old_names and isinstance(old_names[0],str):old_names=[old_names]
    names=copy.deepcopy(old_names);known={n for g in names for n in g}
    added=[n for n,_ in model.named_parameters() if n not in known and n not in old_weights]
    optimizer=copy.deepcopy(saved['optimizer']);next_id=1+max(i for g in optimizer['param_groups'] for i in g['params'])
    group={k:copy.deepcopy(v) for k,v in optimizer['param_groups'][0].items() if k!='params'}
    group['params']=list(range(next_id,next_id+len(added)));optimizer['param_groups'].append(group);names.append(added)
    ema=copy.deepcopy(saved['ema_target_encoder'])
    for k,v in model.core.encoder.state_dict().items():
        key='encoder.'+k
        if key not in ema:ema[key]=v.detach().clone()
    ledger=copy.deepcopy(saved['vjepa_sr_training']);ledger.pop('frozen_parameters_sha256',None)
    dynamics=sum(p.numel() for n,p in model.named_parameters() if n.startswith('vjepa_backend.') and not n.startswith('vjepa_backend.renderer.'))
    report=dict(protocol='vjepa_dynamics300_depth_growth_v1',old_depths=old_depths,encoder_depth=encoder_depth,predictor_depth=predictor_depth,width=model.core.width,dynamics_parameters=dynamics,total_stored_parameters=sum(p.numel() for p in model.parameters()),new_parameter_tensors=len(added),retained_Adam_states=len(optimizer['state']),old_weights_exact=True,new_blocks_identity=True,EMA_old_tensors_preserved=True,created_at=now())
    payload={**saved,'model':model.state_dict(),'config':vars(model.config),'optimizer':optimizer,'optimizer_parameter_names':names,'ema_target_encoder':ema,'vjepa_sr_training':ledger,'vjepa_capacity_growth':report}
    return model,payload,report

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    model,saved=load_world_checkpoint(a.source,'cpu')
    video=torch.rand(1,4,3,model.config.image_size,model.config.image_size);actions=torch.randn(1,4,4);states=torch.randn(1,4,20)
    with torch.inference_mode():
        latents=model.core.encode_video(video);prediction=model.core.predict(latents,actions,states)
    model,payload,report=grow(model,saved)
    with torch.inference_mode():
        grown_latents=model.core.encode_video(video);grown_prediction=model.core.predict(grown_latents,actions,states)
    torch.testing.assert_close(grown_latents,latents,rtol=0,atol=0)
    torch.testing.assert_close(grown_prediction,prediction,rtol=0,atol=0)
    report.update(encoder_output_exact=True,predictor_output_exact=True)
    _save(a.output,payload);atomic_json(a.output.with_suffix('.growth.json'),report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
