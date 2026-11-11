from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import shutil
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from turkeyworld.record_neural_video import (capture_frames, encode_video,
                                             frame_durations, scripted_control)


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Runtime:
    def __init__(self, clock, latencies):
        self.clock, self.latencies = clock, iter(latencies)
        self.state = torch.zeros(1, 20)
        self.memory = torch.zeros(1, 8)
        self.simulation_dt = 1/30
        self.world_metadata = {'training_steps': 2000}
        self.actions = []

    def synchronize(self):
        pass

    def step(self, action):
        self.actions.append(action)
        self.clock.now += next(self.latencies)
        self.state = self.state+1
        self.memory = self.memory+1
        return np.full((16, 16, 3), len(self.actions)*40, dtype=np.uint8)


class RecordNeuralVideoTests(unittest.TestCase):
    def test_actual_latency_is_preserved_and_prior_frame_is_held(self):
        clock = Clock()
        runtime = Runtime(clock, [.05, .20, .05])
        with TemporaryDirectory() as root:
            path = Path(root)/'frames'
            rows, durations, wall = capture_frames(runtime, path, 3, overlay=False,
                                                    clock=clock, sleep=clock.sleep)
            np.testing.assert_allclose([r['ready_wall_seconds'] for r in rows], [.05, .25, .30])
            np.testing.assert_allclose(durations, [.20, .05, 1/30])
            self.assertAlmostEqual(wall, .30)
            self.assertGreater(sum(durations), 3/30)
            self.assertEqual(len(runtime.actions), 3)
            self.assertTrue(torch.equal(runtime.memory, torch.full((1, 8), 3.)))
            for index, row in enumerate(rows):
                np.testing.assert_array_equal(np.asarray(Image.open(path/row['frame'])),
                                              np.full((16, 16, 3), (index+1)*40, dtype=np.uint8))

    def test_fast_inference_keeps_30hz_and_future_engine_is_forbidden(self):
        clock = Clock()
        runtime = Runtime(clock, [.001]*4)
        with TemporaryDirectory() as root:
            with patch('turkeyworld.neural_play.Game.render') as render:
                def predict(action):
                    # A model or policy trying to read future Source pixels is rejected.
                    from turkeyworld.neural_play import Game
                    with self.assertRaisesRegex(AssertionError, 'forbidden'):
                        Game.render(None)
                    return real_step(action)
                real_step = runtime.step
                runtime.step = predict
                rows, durations, _ = capture_frames(runtime, Path(root)/'frames', 4,
                    overlay=False, clock=clock, sleep=clock.sleep)
                render.assert_not_called()
            np.testing.assert_allclose([r['ready_wall_seconds'] for r in rows], np.arange(1, 5)/30)
            np.testing.assert_allclose(durations, np.full(4, 1/30))

    def test_controls_cover_all_requested_directions_and_timing_rejects_invalid(self):
        names = {scripted_control(index, 450)[0] for index in range(450)}
        self.assertTrue({'W', 'S', 'A', 'D', 'Q', 'E', 'W+Q', 'W+E'}.issubset(names))
        self.assertTrue(all(scripted_control(i, 450)[1][3] == 0 for i in range(450)))
        for times in ([0., 0.], [0., float('nan')], []):
            with self.subTest(times=times), self.assertRaises(ValueError):
                frame_durations(times, 1/30)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg/ffprobe not installed')
    def test_real_h264_encoder_repeats_images_at_wall_clock_durations(self):
        with TemporaryDirectory() as root:
            root = Path(root)
            frames = root/'frames'
            frames.mkdir()
            rows = []
            for i, color in enumerate(((240, 0, 0), (0, 240, 0), (0, 0, 240))):
                name = f'frame_{i:06d}.png'
                Image.new('RGB', (16, 16), color).save(frames/name)
                rows.append({'frame': name})
            output = root/'test.mp4'
            encode_video(shutil.which('ffmpeg'), frames, rows, [.1, .2, .1], output)
            result = subprocess.run([shutil.which('ffprobe'), '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name,pix_fmt,width,height,nb_frames,duration',
                '-of', 'json', str(output)], capture_output=True, text=True, check=True)
            stream = json.loads(result.stdout)['streams'][0]
            self.assertEqual(stream['codec_name'], 'h264')
            self.assertEqual(stream['pix_fmt'], 'yuv420p')
            self.assertEqual((stream['width'], stream['height']), (16, 16))
            self.assertAlmostEqual(float(stream['duration']), .4, delta=1/30)
            self.assertGreaterEqual(int(stream['nb_frames']), 11)


if __name__ == '__main__':
    unittest.main()
