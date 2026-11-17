"""Fixed held-out H180 motion panel with the same generated-SR observations as runtime."""
import argparse,json,time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from .models import load_world_checkpoint
from .vjepa_policy_world import VJEPAImaginationAdapter
from .jepa_training_utils import SequenceData
from .rssm_combat_protocol import environment_profile
from .train_vjepa_sr_dynamics import NAV,ENEMY
from .trajectory_objectives import circle_wall_penalty
from .small_enemy_data import digest
from .renderer_student import _hash_tensors
from PIL import Image,ImageDraw,ImageFont
from .training_pipeline import atomic_json,now
from .workspace_paths import ROOT
from .rssm_drift_preview import build_drift_preview
from .structured_state import player_command_delta

def configure_memory_mode(model,mode):
    """Read-only architecture ablation; parameter values/shapes stay unchanged."""
    if mode=='checkpoint':return
    if mode not in ('pooled','spatial'):raise ValueError('Unknown validation memory mode')
    pooled=mode=='pooled';core=model.vjepa_backend.core
    core.pooled_memory=pooled;core.visual_tokens=1 if pooled else core.encoder.tokens_per_frame
    core.block_tokens=core.visual_tokens+2;core.memory_tokens=core.context_frames*core.block_tokens
    core.predictor.grid=1 if pooled else core.encoder.grid;core.predictor.visual_tokens=core.visual_tokens;core.predictor.block_tokens=core.block_tokens
    for value in (model,model.vjepa_backend):value.config=replace(value.config,vjepa_pooled_memory=pooled)

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--world',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--preview-dir',type=Path);p.add_argument('--memory-mode',choices=['checkpoint','pooled','spatial'],default='checkpoint');p.add_argument('--feedback-config',type=Path);p.add_argument('--feedback-mode',choices=['on','off'],default='on');p.add_argument('--observation-domain',choices=['sr','source'],default='sr');a=p.parse_args();torch.set_num_threads(2)
    if a.observation_domain=='source' and (a.feedback_mode!='off' or a.preview_dir is not None):p.error('Source-context diagnostic requires feedback off and no production preview')
    model,saved=load_world_checkpoint(a.world,'cuda');step=saved['training_steps'];del saved
    configure_memory_mode(model,a.memory_mode)
    if a.feedback_config:model.vjepa_sr_feedback=json.loads(a.feedback_config.read_text())
    if a.observation_domain=='source':model.vjepa_sr_feedback=None
    model.eval().requires_grad_(False);adapter=VJEPAImaginationAdapter(model)
    if a.feedback_mode=='off' and adapter.feedback_settings:adapter.feedback_settings={**adapter.feedback_settings,'every':10**15}

    reference=json.loads((ROOT/'training_runs/rssm_moe_nav15_enemy25_cap20_20261113/rssm_validation_latest.json').read_text())
    report=dict(protocol='vjepa_SR_motion_heldout_v1',step=step,horizon=180,Source_future_feedback=False,Source_RGB_encoder_inputs=a.observation_domain=='source',observation_domain=a.observation_domain,source_held_out=True,generated_at=now(),memory_tokens=model.core.memory_tokens,visual_tokens=model.core.visual_tokens,feedback_mode=a.feedback_mode)
    preview_rows=[];map_drifts={};snapshots=[1,60,120,180]
    for stream,folder in [('navigation',NAV),('enemy_motion',ENEMY)]:
        data=SequenceData(folder.parent/'val',split='val',image_size=32,mode='stream',device='cpu');windows=reference[stream]['panel']['windows']
        errors=[];npc=[];walls=[];files=[];per_window=[];offset_errors={str(t):dict(player=[],enemy=[]) for t in [1,8,16,48,112,180]};speeds=[];kinematics=[]
        with environment_profile(model,navigation_only=stream=='navigation'):
            for first in range(0,len(windows),2):
                group=windows[first:first+2];batch=[];past=[];past_rgb=[]
                for row in group:
                    ep=next(i for i,e in enumerate(data.episodes) if Path(e['path']).name==row['episode']);episode=data.episodes[ep];start=row['start']
                    batch.append(data.batch(1,context=1,horizon=180,indices=np.array([[ep,start]])))
                    past.append(torch.tensor(np.asarray(episode['states'][[max(0,start-3+i) for i in range(4)]]),device='cuda'))
                    if a.observation_domain=='source':
                        indexes=[max(0,start-3+i) for i in range(4)]
                        past_rgb.append(torch.tensor(np.asarray(episode['frames'][indexes]),device='cuda').permute(0,3,1,2).float()/255)
                    files.append(dict(path=episode['path'],sha256=digest(episode['path']),start=start))
                states=torch.stack(past);grid=torch.cat([b['map'] for b in batch]).to('cuda');rgb=torch.stack(past_rgb) if past_rgb else torch.zeros(len(group),4,3,32,32,device='cuda')
                current=adapter.initialize(states,grid,rgb);initial_state=current.state.detach().clone();actions=torch.cat([b['actions'] for b in batch]).to('cuda');target=torch.cat([b['target_states'] for b in batch]).to('cuda');points=[]
                for tick in range(180):current=adapter.transition_prior(current,actions[:,tick]);points.append(current.state)
                predicted=torch.stack(points,1)
                if a.preview_dir is not None and first==0:
                    drift=build_drift_preview(batch[0],dict(context=1,horizon=180,display_offsets=snapshots),torch.cat((initial_state[:1],predicted[0]),0),step=step)
                    drift['kind']='vjepa_preview_map_drift_v1';drift['model']='V-JEPA pooled SR feedback';map_drifts[stream]=drift
                    references=[];generated=[]
                    for tick in snapshots:
                        references.append(batch[0]['frames'][0,tick].cpu())
                        generated.append(adapter.feedback_renderer.image(predicted[:1,tick-1],grid[:1],navigation_only=stream=='navigation')[0].cpu())
                    preview_rows.append(('NAV Source32' if stream=='navigation' else 'NPC Source32',references))
                    preview_rows.append(('NAV V-JEPA SR128' if stream=='navigation' else 'NPC V-JEPA SR128',generated))
                errors.extend(((predicted[:,-1,:2]-target[:,-1,:2])*16).norm(dim=-1).cpu().tolist())
                if stream=='enemy_motion':npc.extend(((predicted[:,-1,8:10]-target[:,-1,8:10])*16).norm(dim=-1).cpu().tolist())
                walls.extend((circle_wall_penalty(predicted[...,:2]*16,grid,.20)>0).sum(1).cpu().tolist())
                previous=torch.cat((initial_state[:,None],predicted[:,:-1]),1)
                source_previous=torch.cat((torch.cat([b['states'][:,:1] for b in batch]).to('cuda'),target[:,:-1]),1)
                command=player_command_delta(previous.flatten(0,1),actions.flatten(0,1)).reshape(len(group),180,2)*16
                source_command=player_command_delta(source_previous.flatten(0,1),actions.flatten(0,1)).reshape(len(group),180,2)*16
                own_delta=(predicted[...,:2]-previous[...,:2])*16;source_delta=(target[...,:2]-source_previous[...,:2])*16
                norm=own_delta.norm(dim=-1);command_norm=command.norm(dim=-1);source_norm=source_delta.norm(dim=-1)
                moving=command_norm>1e-6;idle=~moving;nonzero=moving&(norm>1e-6)
                dot=(own_delta*command).sum(-1)
                kinematics.append(dict(moving=int(moving.sum()),idle=int(idle.sum()),backward=int(((dot< -1e-7)&moving).sum()),
                    idle_speed_sum=float((norm[idle]*30).sum()),excess=int(((norm-command_norm)*30>.01).sum()),
                    source_excess=int(((source_norm-source_command.norm(dim=-1))*30>.01).sum()),steps=norm.numel(),
                    direction_cosine_sum=float((dot[nonzero]/(norm[nonzero]*command_norm[nonzero])).sum()),direction_count=int(nonzero.sum())))

                for tick,values in offset_errors.items():
                    index=int(tick)-1
                    values['player'].extend(((predicted[:,index,:2]-target[:,index,:2])*16).norm(dim=-1).cpu().tolist())
                    if stream=='enemy_motion':values['enemy'].extend(((predicted[:,index,8:10]-target[:,index,8:10])*16).norm(dim=-1).cpu().tolist())
                for actor,row in enumerate(group):
                    own_speed=((predicted[actor,:,:2]-previous[actor,:,:2])*480).norm(dim=-1).mean()
                    source_speed=((target[actor,:,:2]-source_previous[actor,:,:2])*480).norm(dim=-1).mean()
                    speeds.append((float(own_speed),float(source_speed)))
                    per_window.append(dict(episode=row['episode'],start=row['start'],player_error_tiles=float(((predicted[actor,-1,:2]-target[actor,-1,:2])*16).norm()),
                        enemy_error_tiles=float(((predicted[actor,-1,8:10]-target[actor,-1,8:10])*16).norm()) if stream=='enemy_motion' else None,
                        player_speed_tiles_s=float(own_speed),Source_player_speed_tiles_s=float(source_speed),
                        player_final_error_xy_tiles=((predicted[actor,-1,:2]-target[actor,-1,:2])*16).cpu().tolist()))

        report[stream]=dict(count=len(windows),player_XY_H180_mean_tiles=float(np.mean(errors)),enemy_XY_H180_mean_tiles=float(np.mean(npc)) if npc else None,
            player_wall_overlap_ticks_mean=float(np.mean(walls)),windows=files,per_window=per_window,
            offsets={k:dict(player_XY_mean_tiles=float(np.mean(v['player'])),enemy_XY_mean_tiles=float(np.mean(v['enemy'])) if v['enemy'] else None) for k,v in offset_errors.items()},
            player_speed_tiles_s=float(np.mean([v[0] for v in speeds])),Source_player_speed_tiles_s=float(np.mean([v[1] for v in speeds])),
            kinematics=dict(backward_command_fraction=sum(v['backward'] for v in kinematics)/max(1,sum(v['moving'] for v in kinematics)),
                movement_on_idle_mean_speed_tiles_s=sum(v['idle_speed_sum'] for v in kinematics)/sum(v['idle'] for v in kinematics) if sum(v['idle'] for v in kinematics) else None,
                idle_command_steps=sum(v['idle'] for v in kinematics),
                speed_limit_exceeded_fraction=sum(v['excess'] for v in kinematics)/sum(v['steps'] for v in kinematics),
                Source_speed_limit_exceeded_fraction=sum(v['source_excess'] for v in kinematics)/sum(v['steps'] for v in kinematics),
                mean_direction_cosine=sum(v['direction_cosine_sum'] for v in kinematics)/max(1,sum(v['direction_count'] for v in kinematics))))
        atomic_json(a.output,report)
    if a.preview_dir is not None:
        root=a.preview_dir;root.mkdir(parents=True,exist_ok=True)
        canvas=Image.new('RGB',(1024,1152),(13,19,24));draw=ImageDraw.Draw(canvas)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
        for row,(title,images) in enumerate(preview_rows):
            for col,pixels in enumerate(images):
                image=Image.fromarray((pixels.clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8'))
                canvas.paste(image.resize((256,256),Image.Resampling.NEAREST),(col*256,row*288+30))
                draw.text((col*256+6,row*288+6),title+' | '+str(snapshots[col]),font=font,fill='white')
        temp=root/'preview.tmp.png';canvas.save(temp);temp.replace(root/'preview.png')
        metadata=dict(kind='vjepa_SR_feedback_rollout_v1',step=step,horizon=180,row_count=4,
            source_held_out=True,Source_future_feedback=False,Source_RGB_encoder_inputs=False,
            generated_at=now(),every=10,model_sha256=_hash_tensors(model.state_dict()),
            image_sha256=digest(root/'preview.png'),reference_resolution=32,generated_resolution=128,
            columns=snapshots,display_offsets=snapshots,context=1,observed_context_frames=4,generated_rollout_transitions=180,
            map_drift=map_drifts.get('enemy_motion'),map_drifts=map_drifts,feedback=model.vjepa_sr_feedback)
        atomic_json(root/'preview.json',metadata);atomic_json(root/'vjepa_validation.json',report)
        atomic_json(root/'validation'/f'vjepa_sr_{step:06d}.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
