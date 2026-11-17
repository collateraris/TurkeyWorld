"""Original LEFT, restored V-JEPA CENTRE, RSSM RIGHT; one shared action tape."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .neural_play import NeuralRuntime,_arguments
from .compare_runtime import PairedRuntime
from .compare_play import keyboard_action,shortcut_key,_surface,_font,_work_area
from .training_pipeline import atomic_json,now


def main():
    p=argparse.ArgumentParser()
    for key in ('vjepa','rssm','render-world','renderer','sr','session-dir'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--two-panel',action='store_true');p.add_argument('--steps',type=int,default=180);p.add_argument('--headless',action='store_true');a=p.parse_args()
    a.session_dir.mkdir(parents=True,exist_ok=True)
    pygame=None
    if not a.headless:
        import pygame
        pygame.init();w,h=_work_area(pygame);columns=2 if a.two_panel else 3;panel=min(768 if a.two_panel else 640,(w-64)//columns,h-210)
        screen=pygame.display.set_mode((columns*panel+12*(columns+1),panel+200));pygame.display.set_caption('TurkeyWorld | Source / V-JEPA 2-AC' if a.two_panel else 'TurkeyWorld | Source / V-JEPA 2-AC / RSSM')
        screen.fill((13,19,24));pygame.display.flip()
        atomic_json(a.session_dir/'status.json',dict(status='loading',ready=False,window_ready=True))
    common=['--sr',str(a.sr),'--device','cuda','--display-size','1024','--smooth','--threads','2','--warmup','0']
    right_args=_arguments(['--world',str(a.rssm),'--renderer-checkpoint',str(a.renderer),'--allow-unqualified-codec','--base-only',*common]);right_args.initial_navigation_only=True
    right=None if a.two_panel else NeuralRuntime(right_args)
    # The render bridge has its own calibrated map/geometry. Dynamics retain
    # their original map encoders; no RSSM transition advances for V-JEPA.
    render_args=_arguments(['--world',str(a.render_world),'--renderer-checkpoint',str(a.renderer),'--allow-unqualified-codec','--base-only',*common]);render_args.initial_navigation_only=True
    render=NeuralRuntime(render_args)
    centre_args=_arguments(['--world',str(a.vjepa),*common]);centre_args.initial_navigation_only=True
    centre=NeuralRuntime(centre_args)
    original_render=render.model.render
    def common_render(state,features,map_features):
        # Explicit snapshot feature policy avoids mixing incompatible auxiliary
        # feature spaces and gives both dynamics exactly the same renderer.
        return original_render(state,torch.zeros(len(state),128,device=state.device),render.map_features)
    for runtime in ([centre] if a.two_panel else [centre,right]):
        runtime.model.render=common_render
        old=runtime.model.flow_renderer
        holder=nn.Module();holder.kind='base_only';holder.steps=0
        if hasattr(old,'geometry_opacity'):holder.geometry_opacity=old.geometry_opacity
        runtime.model.flow_renderer=holder
    left_pair=PairedRuntime(centre,source_size=256,horizon=a.steps)
    right_pair=left_pair if a.two_panel else PairedRuntime(right,source_size=256,horizon=a.steps)
    if right is not None and not torch.equal(centre.initial_state,right.initial_state):raise ValueError('Dynamics did not share the same Source initialization')
    paused=True;complete=False;running=True;physical=set();next_tick=time.perf_counter()
    def frames():return (left_pair.source_frame,left_pair.neural_frame) if a.two_panel else (right_pair.source_frame,left_pair.neural_frame,right_pair.neural_frame)
    def status():
        atomic_json(a.session_dir/'status.json',dict(application='three_way_source_vjepa_rssm',status='complete_paused' if complete else 'paused' if paused else 'running',
            ready=True,window_ready=not a.headless,steps=right_pair.steps,horizon=a.steps,
            panels=['Source','V-JEPA 2-AC'] if a.two_panel else ['Source','V-JEPA 2-AC','RSSM'],vjepa_step=centre.world_metadata['training_steps'],rssm_step=right.world_metadata['training_steps'] if right is not None else None,
            vjepa_sr_feedback=centre.feedback_settings,vjepa_visual_tokens=centre.model.vjepa_backend.core.visual_tokens,
            renderer='Base31709 without RF/VQGAN',appearance_policy='shared_snapshot_zero128',source_future_feedback=False,
            vjepa_player_drift=left_pair.status().get('xy_drift_tiles'),rssm_player_drift=right_pair.status().get('xy_drift_tiles'),
            updated_at=now()))
    def advance(action):
        left_pair.step(action)
        if not a.two_panel:right_pair.step(action)
        if not np.array_equal(left_pair.source.state_vector(),right_pair.source.state_vector()):raise ValueError('Source reference replicas diverged')
    if a.headless:
        for _ in range(a.steps):advance([.7,.2,.1,0.])
        complete=True;status();return
    font=_font(pygame,22);small=_font(pygame,17);clock=pygame.time.Clock();status()
    try:
        while running:
            for event in pygame.event.get():
                if event.type==pygame.QUIT:running=False
                elif event.type==pygame.WINDOWFOCUSLOST:physical.clear()
                elif event.type==pygame.KEYUP:physical.discard(event.scancode)
                elif event.type==pygame.KEYDOWN:
                    physical.add(event.scancode);key=shortcut_key(pygame,event)
                    if key==pygame.K_ESCAPE:running=False
                    elif key in (pygame.K_RETURN,pygame.K_r):
                        if complete or key==pygame.K_r:left_pair.reset();right_pair.reset() if not a.two_panel else None;complete=False
                        paused=False;next_tick=time.perf_counter()
                    elif key==pygame.K_p:paused=not paused
                    elif key==pygame.K_t:
                        navigation=right_pair.status()['environment']=='combat'
                        left_pair.set_environment('navigation' if navigation else 'combat')
                        if not a.two_panel:right_pair.set_environment('navigation' if navigation else 'combat')
                        render.model.config=__import__('dataclasses').replace(render.model.config,navigation_only=navigation)
                        paused=False;complete=False;next_tick=time.perf_counter()
            if not paused and not complete and time.perf_counter()>=next_tick:
                started=time.perf_counter();advance(keyboard_action(pygame,pygame.key.get_pressed(),physical));next_tick=max(started+1/30,time.perf_counter())
                if right_pair.steps>=a.steps:complete=True;paused=True
                status()
            screen.fill((13,19,24))
            for i,(title,pixels) in enumerate(zip(('Оригинал','V-JEPA 2-AC · справа') if a.two_panel else ('Оригинал','V-JEPA 2-AC · центр','RSSM · справа'),frames())):
                x=12+i*(panel+12);screen.blit(font.render(title,True,(235,239,243)),(x,10));screen.blit(_surface(pygame,pixels,panel),(x,70))
                screen.blit(small.render('Source256' if i==0 else 'Общий Base → SR · без Flow',True,(163,180,187)),(x,40))
            y=panel+83;label='ПАУЗА · Enter — начать' if paused and not complete else 'ПРОБА ЗАВЕРШЕНА · R — заново' if complete else 'Общие команды, независимая динамика'
            screen.blit(small.render(f'{label} · шаг {right_pair.steps}/{a.steps}',True,(235,239,243)),(12,y))
            screen.blit(small.render('WASD — движение · Q/E — поворот · Space — огонь · T — бой/без боя · P — пауза · Esc — выход',True,(163,180,187)),(12,y+30))
            pygame.display.flip();clock.tick(60)
    finally:status();pygame.quit()

if __name__=='__main__':main()
