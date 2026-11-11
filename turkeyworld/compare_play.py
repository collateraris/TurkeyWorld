"""One keyboard and one30Hz clock for side-by-side Source/neural playback.

The Source side waits for every neural update. This GUI never advances either
side independently or corrects neural rollout states automatically. T is an
explicit navigation/combat scene event. Opening is paused, and completing the
450-step trial holds the window until restart/quit.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import torch

from .neural_play import NeuralRuntime, _arguments as runtime_arguments
from .training_pipeline import atomic_json


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel-size', type=int, default=0,
                        help='Panel size in pixels;0 chooses up to768 and fits the desktop work area.')
    parser.add_argument('--source-size', type=int, choices=(32, 128, 256), default=256,
                        help='Actual classical Source render resolution, separately labeled.')
    parser.add_argument('--comparison-mode', choices=('autonomous', 'shared_camera'), default='autonomous',
                        help='shared_camera uses the same Source pose for both displayed images; RSSM stays independent.')
    parser.add_argument('--shared-camera-entities', choices=('source', 'neural'), default='neural',
                        help='neural shares only the player camera; NPCs, health and combat remain RSSM predictions.')
    parser.add_argument('--initial-environment', choices=('navigation', 'combat'), default='navigation',
                        help='Start without enemies for navigation; T toggles one-enemy combat.')
    parser.add_argument('--session-dir', type=Path,
                        default=Path('training_runs')/('compare_play_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    gui, remaining = parser.parse_known_args(argv)
    neural = runtime_arguments(remaining)
    if gui.panel_size and not 128 <= gui.panel_size <= 2048:
        parser.error('--panel-size must be0 or128..2048')
    if neural.two_stage_upscale or neural.display_size != 1024 or neural.sr is None or not neural.smooth:
        parser.error('Comparison requires --sr --display-size1024 --smooth; omit --two-stage-upscale')
    if neural.headless:
        parser.error('Comparison opens one visible window; omit --headless')
    if neural.steps not in (0, 450):
        parser.error('The paired trial is450 steps /15 simulated seconds')
    neural.steps = 450
    # PairedRuntime renders the cached initial pose without consuming a tick.
    # No discarded warmup transitions belong to either trial's reported clock.
    neural.warmup = 0
    neural.initial_navigation_only = gui.initial_environment == 'navigation'
    return gui, neural


def panel_layout(work_width, work_height, requested=0):
    """Fit both equally sized panels and readable controls within one work area."""
    if min(work_width, work_height) < 400:
        raise ValueError('Desktop work area is too small for a paired comparison')
    margin, gap, header, footer = 12, 12, 88, 112
    panel = min(requested or 768, (work_width-2*margin-gap-40)//2,
                work_height-header-footer-40)
    if panel < 128:
        raise ValueError('Desktop work area cannot fit two128px panels')
    return {'panel': int(panel), 'width': int(2*panel+2*margin+gap),
            'height': int(panel+header+footer), 'left': (margin, header),
            'right': (margin+panel+gap, header), 'header': header, 'footer': footer}


def _work_area(pygame):
    if sys.platform == 'win32':
        rect = wintypes.RECT()
        user32 = ctypes.WinDLL('user32', use_last_error=True)
        user32.SystemParametersInfoW.argtypes = [wintypes.UINT, wintypes.UINT, ctypes.c_void_p, wintypes.UINT]
        user32.SystemParametersInfoW.restype = wintypes.BOOL
        if user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
            return rect.right-rect.left, rect.bottom-rect.top
    sizes = pygame.display.get_desktop_sizes()
    return sizes[0] if sizes else (1366, 768)


def keyboard_action(pygame, keys):
    """Exactly one shared action is polled for a paired simulation tick."""
    return [float(keys[pygame.K_w] or keys[pygame.K_UP])-float(keys[pygame.K_s] or keys[pygame.K_DOWN]),
            float(keys[pygame.K_d])-float(keys[pygame.K_a]),
            float(keys[pygame.K_e] or keys[pygame.K_RIGHT])-float(keys[pygame.K_q] or keys[pygame.K_LEFT]),
            float(keys[pygame.K_SPACE] or pygame.mouse.get_pressed(3)[0])]


def switch_test_environment(pair):
    """T switches the scene; an exhausted trial first gets an explicit restart."""
    target = 'combat' if pair.status()['environment'] == 'navigation' else 'navigation'
    if pair.finished:
        pair.reset()
    return pair.set_environment(target)


def _font(pygame, size):
    windows_font = Path('C:/Windows/Fonts/segoeui.ttf')
    filename = str(windows_font) if windows_font.is_file() else pygame.font.match_font('dejavusans,arial')
    return pygame.font.Font(filename, size)


def _surface(pygame, image, panel):
    surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    return pygame.transform.smoothscale(surface, (panel, panel))


def interactive(pair, gui, args, pygame, screen, layout):
    """Drive the paired core; Enter/P/R affect both sides at trial boundaries."""
    font = _font(pygame, max(16, min(25, layout['panel']//30)))
    small = _font(pygame, max(14, min(20, layout['panel']//38)))
    color, muted, orange = (235, 239, 243), (163, 180, 187), (255, 198, 115)
    clock = pygame.time.Clock()
    frames = (_surface(pygame, pair.source_frame, layout['panel']),
              _surface(pygame, pair.neural_frame, layout['panel']))
    paused, trial_complete, running = True, False, True
    started = time.perf_counter()
    next_update = started
    last_status = None
    displayed = 0
    def status(phase, *, force=False):
        nonlocal last_status
        metrics = pair.status()
        value = (phase, metrics['steps'])
        if not force and value == last_status:
            return
        report = {'application': 'paired_source_neural_game', 'status': phase,
            'ready': True, 'window_ready': True, 'paused': paused,
            'trial_complete': trial_complete, 'trial_steps': args.steps,
            'window_size': [layout['width'], layout['height']], 'panel_size': layout['panel'],
            'source_render_size': gui.source_size,
            'neural_display_chain': 'native32 -> learnedSR128 -> bilinear1024 -> fitted panel',
            'one_keyboard_action_per_paired_tick': True, 'independent_source_clock': False,
            'catchup_steps': 0, 'trial_holds_after_completion': True,
            'updated_at': datetime.now(timezone.utc).isoformat(), **metrics}
        atomic_json(gui.session_dir/'status.json', report)
        last_status = value
    status('paused', force=True)
    print('READY PAUSED | ENTER start | T navigation/combat | E turn right | P pause | R reset BOTH | ESC quit', flush=True)
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        pair.reset()
                        frames = (_surface(pygame, pair.source_frame, layout['panel']),
                                  _surface(pygame, pair.neural_frame, layout['panel']))
                        paused, trial_complete = False, False
                        next_update = time.perf_counter()
                        status('running', force=True)
                    elif event.key == pygame.K_RETURN and not trial_complete:
                        paused = False
                        next_update = time.perf_counter()
                        status('running')
                    elif event.key == pygame.K_p and not trial_complete:
                        paused = not paused
                        next_update = time.perf_counter()
                        status('paused' if paused else 'running')
                    elif event.key == pygame.K_t:
                        source_rgb, neural_rgb = switch_test_environment(pair)
                        frames = (_surface(pygame, source_rgb, layout['panel']),
                                  _surface(pygame, neural_rgb, layout['panel']))
                        paused, trial_complete = False, False
                        next_update = time.perf_counter()
                        status('running', force=True)
                    elif event.key == pygame.K_TAB:
                        mode = pair.status().get('mode', 'autonomous')
                        pair.set_mode('autonomous' if mode == 'shared_camera' else 'shared_camera')
                        frames = (_surface(pygame, pair.source_frame, layout['panel']),
                                  _surface(pygame, pair.neural_frame, layout['panel']))
                        status('complete_paused' if trial_complete else 'paused' if paused else 'running', force=True)
            if not running or args.seconds and time.perf_counter()-started >= args.seconds:
                break
            if not paused and time.perf_counter() >= next_update:
                update_started = time.perf_counter()
                action = keyboard_action(pygame, pygame.key.get_pressed())
                source_rgb, neural_rgb = pair.step(action)
                frames = (_surface(pygame, source_rgb, layout['panel']),
                          _surface(pygame, neural_rgb, layout['panel']))
                # Never accumulate backlog or run a second Source tick alone.
                next_update = max(update_started+pair.neural.simulation_dt, time.perf_counter())
                if pair.steps >= args.steps:
                    paused, trial_complete = True, True
                status('complete_paused' if trial_complete else 'running')
            metrics = pair.status()
            steps = metrics['steps']
            drift = metrics.get('xy_drift_tiles')
            drift_text = f'{drift:.3f}' if isinstance(drift, (int, float)) else '—'
            state_label = 'ПРОБА ЗАВЕРШЕНА · R — начать заново' if trial_complete else 'ПАУЗА · ENTER — начать / продолжить' if paused else 'ОДИНАКОВЫЕ КОМАНДЫ · ОБЩИЕ ШАГИ'
            mode_label = (('ОБЩАЯ КАМЕРА · NPC и бой предсказывает RSSM' if gui.shared_camera_entities == 'neural'
                           else 'СОСТОЯНИЯ SOURCE · только проверка рендера') if metrics.get('mode') == 'shared_camera'
                          else 'АВТОНОМНАЯ ДИНАМИКА · одинаковые команды')
            environment_label = 'БОЙ · 1 ПРОТИВНИК' if metrics.get('environment') == 'combat' else 'НАВИГАЦИЯ · БЕЗ ПРОТИВНИКОВ'
            screen.fill((13, 19, 24))
            screen.blit(font.render('Классическая игра · СЛЕВА', True, color), (12, 8))
            screen.blit(font.render('Нейросетевая игра · СПРАВА', True, color), (layout['right'][0], 8))
            screen.blit(small.render(f'Source {gui.source_size}×{gui.source_size} · обычный движок', True, muted), (12, 42))
            screen.blit(small.render(environment_label, True, orange), (12, 63))
            screen.blit(small.render('Нейро32 → SR128 → bilinear1024 · эксперимент', True, muted), (layout['right'][0], 40))
            screen.blit(small.render(mode_label, True, orange), (layout['right'][0], 63))
            screen.blit(frames[0], layout['left'])
            screen.blit(frames[1], layout['right'])
            y = layout['header']+layout['panel']+10
            screen.blit(small.render(f'{state_label} | шаг {steps}/{args.steps} | симуляция {steps*pair.neural.simulation_dt:.2f} с | XY-разница {drift_text} клетки', True, orange), (12, y))
            screen.blit(small.render('W/S, ↑/↓ — ходьба · A/D — стрейф · Q/E, ←/→ — поворот · SPACE/ЛКМ — выстрел', True, color), (12, y+30))
            screen.blit(small.render('T — бой / навигация · ENTER — старт · P — пауза · R — заново · TAB — камера · ESC — выход', True, muted), (12, y+58))
            pygame.display.set_caption(f'TurkeyWorld COMPARE | Source LEFT / Neural RIGHT | {metrics.get("environment", "navigation")} | {metrics.get("mode", "autonomous")} | updates {steps}/{args.steps} | '+('PAUSED' if paused else 'RUNNING'))
            pygame.display.flip()
            if displayed == 0:
                pygame.image.save(screen, str(gui.session_dir/'window_ready.png'))
            displayed += 1
            clock.tick(args.fps)
    finally:
        status('closed', force=True)
    return {'wall_seconds': time.perf_counter()-started, 'displayed_frames': displayed, **pair.report()}


def main(argv=None):
    gui, args = arguments(argv)
    gui.session_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(gui.session_dir/'status.json', {'status': 'startup', 'ready': False, 'window_ready': False})
    # CUDA must initialize before large checkpoint loading on this Windows setup.
    if args.device == 'cuda':
        torch.cuda.init()
        torch.empty(1, device='cuda')
    import pygame
    from .compare_runtime import PairedRuntime
    pygame.init()
    layout = panel_layout(*_work_area(pygame), gui.panel_size)
    screen = pygame.display.set_mode((layout['width'], layout['height']))
    pygame.display.set_caption('TurkeyWorld COMPARE | loading Source + Neural')
    font = _font(pygame, 24)
    screen.fill((13, 19, 24))
    screen.blit(font.render('Загружаем классическую и нейросетевую игру…', True, (235, 239, 243)), (24, 28))
    pygame.display.flip()
    atomic_json(gui.session_dir/'status.json', {'status': 'startup', 'ready': False, 'window_ready': True,
                                               'window_size': [layout['width'], layout['height']]})
    try:
        neural = NeuralRuntime(args)
        pair = PairedRuntime(neural, source_size=gui.source_size, horizon=args.steps,
                             shared_camera_entities=gui.shared_camera_entities)
        pair.set_mode(gui.comparison_mode)
        report = interactive(pair, gui, args, pygame, screen, layout)
        atomic_json(gui.session_dir/'report.json', report)
    except Exception as error:
        atomic_json(gui.session_dir/'status.json', {'status': 'failed', 'ready': False,
                                                   'error': f'{type(error).__name__}: {error}'})
        raise
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()
