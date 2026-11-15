"""Launcher help/dry-runs use a fake interpreter, never a training worker."""
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest

ROOT=Path(__file__).resolve().parents[1]
BASH=Path('C:/Program Files/Git/bin/bash.exe')if os.name=='nt'else Path(shutil.which('bash')or'/no-bash')


@unittest.skipUnless(BASH.is_file(),'Bash is unavailable')
class TrainingLauncherTests(unittest.TestCase):
    def test_help_has_no_training_or_inhibitor_side_effects(self):
        result=subprocess.run([str(BASH),'scripts/run_training.sh','--help'],cwd=ROOT,
            capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--module',result.stdout);self.assertIn('tmux',result.stdout)

    def test_allow_sleep_forwards_exact_generic_command_to_fake_python(self):
        with TemporaryDirectory(dir=ROOT)as directory:
            fake=Path(directory)/'fake-python'
            fake.write_text('#!/usr/bin/env bash\nprintf \'<%s>\\n\' "$@"\n',encoding='utf-8',newline='\n')
            if os.name!='nt':fake.chmod(0o755)
            relative=fake.relative_to(ROOT).as_posix()
            environment=dict(os.environ,TURKEYWORLD_PYTHON='./'+relative)
            result=subprocess.run([str(BASH),'scripts/run_training.sh','--allow-sleep','--module','target.module',
                '--value','a path with spaces'],cwd=ROOT,env=environment,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.splitlines(),['<-m>','<turkeyworld.bootstrap>','<--module>',
            '<target.module>','<--value>','<a path with spaces>'])
        self.assertIn('explicitly disabled',result.stderr)


if __name__=='__main__':unittest.main()
