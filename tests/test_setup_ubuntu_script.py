"""Installer syntax/dry-run only: no package install, Python/GPU or training run."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
GIT_BASH = Path('C:/Program Files/Git/bin/bash.exe')
BASH = str(GIT_BASH) if sys.platform == 'win32' and GIT_BASH.exists() else (
    shutil.which('bash') if sys.platform != 'win32' else None)


@unittest.skipUnless(BASH, 'A Bash parser is required; the Windows WSL launcher is not used')
class UbuntuInstallerTests(unittest.TestCase):
    def invoke(self, *arguments):
        return subprocess.run([BASH, 'scripts/setup_ubuntu.sh', *arguments], cwd=ROOT,
            text=True, encoding='utf-8', capture_output=True, check=False)

    def test_bash_syntax(self):
        result = subprocess.run([BASH, '-n', 'scripts/setup_ubuntu.sh'], cwd=ROOT,
            capture_output=True, text=True, encoding='utf-8', check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dry_run_never_executes_missing_python_or_creates_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'env with spaces'
            result = self.invoke('--dry-run', '--python', '__TW_MUST_NOT_EXECUTE__', '--venv', str(target))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('torch==2.11.0+cu128', result.stdout)
            self.assertIn('torchvision==0.26.0+cu128', result.stdout)
            self.assertIn('https://download.pytorch.org/whl/cu128', result.stdout)
            self.assertIn('No training was started', result.stdout)
            self.assertFalse(target.exists())
        cpu = self.invoke('--dry-run', '--cpu', '--python', '__TW_MUST_NOT_EXECUTE__')
        self.assertEqual(cpu.returncode, 0, cpu.stderr)
        self.assertIn('torch==2.11.0+cpu', cpu.stdout)
        self.assertIn('.venv-ubuntu-cpu', cpu.stdout)

    def test_unknown_options_and_missing_values_are_rejected(self):
        self.assertEqual(self.invoke('--unknown').returncode, 2)
        self.assertEqual(self.invoke('--venv').returncode, 2)


if __name__ == '__main__':
    unittest.main()
