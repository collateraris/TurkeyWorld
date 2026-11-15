"""Portable startup and process inspection; no CUDA initialization or workers."""
from contextlib import redirect_stdout,redirect_stderr
import io
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from turkeyworld import bootstrap
from turkeyworld import dreamer_motion_cycle as cycle


class PortableProcessTests(unittest.TestCase):
    def test_posix_live_missing_permission_denied_and_zombie_pids(self):
        with patch.object(cycle.sys,'platform','linux'),patch.object(cycle.os,'kill')as signal,\
                patch.object(cycle.Path,'read_text',return_value='123 (process (name) with spaces) S 12 0'):
            self.assertTrue(cycle._posix_alive(123))
            signal.assert_called_once_with(123,0)
        with patch.object(cycle.os,'kill',side_effect=ProcessLookupError):
            self.assertFalse(cycle._posix_alive(123))
        with patch.object(cycle.sys,'platform','linux'),patch.object(cycle.os,'kill',side_effect=PermissionError),\
                patch.object(cycle.Path,'read_text',side_effect=PermissionError):
            self.assertTrue(cycle._posix_alive(123))
        for state in('Z','X','x'):
            with self.subTest(state=state),patch.object(cycle.sys,'platform','linux'),\
                    patch.object(cycle.os,'kill'),\
                    patch.object(cycle.Path,'read_text',return_value=f'123 (unreaped child) {state} 12 0'):
                self.assertFalse(cycle._posix_alive(123))

    def test_posix_fallback_when_proc_is_unmounted_and_strict_invalid_pids(self):
        with patch.object(cycle.sys,'platform','linux'),patch.object(cycle.os,'kill'),\
                patch.object(cycle.Path,'read_text',side_effect=FileNotFoundError):
            self.assertTrue(cycle._posix_alive(123))
        with patch.object(cycle,'_posix_alive',return_value=True)as posix,patch.object(cycle.os,'name','posix'):
            self.assertTrue(cycle.alive(123));posix.assert_called_once_with(123)
            for pid in(None,False,True,0,-1,1.5,'123'):
                with self.subTest(pid=pid):self.assertFalse(cycle.alive(pid))
            self.assertEqual(posix.call_count,1)

    def test_windows_handle_is_closed_and_posix_flags_are_zero(self):
        kernel=SimpleNamespace(OpenProcess=Mock(return_value=99),
            WaitForSingleObject=Mock(return_value=258),CloseHandle=Mock())
        with patch.object(cycle.os,'name','nt'),\
                patch.object(cycle.ctypes,'WinDLL',return_value=kernel,create=True):
            self.assertTrue(cycle.alive(123))
        kernel.CloseHandle.assert_called_once_with(99)
        kernel.WaitForSingleObject.assert_called_once_with(99,0)
        with patch.object(cycle.os,'name','posix'),\
                patch.object(cycle.subprocess,'CREATE_NO_WINDOW',123456,create=True):
            self.assertEqual(cycle.child_process_options(),{'creationflags':0})
        with patch.object(cycle.os,'name','nt'),\
                patch.object(cycle.subprocess,'CREATE_NO_WINDOW',123456,create=True):
            self.assertEqual(cycle.child_process_options(),{'creationflags':123456})

    def test_controller_launch_uses_current_python_portable_module_and_exact_arguments(self):
        controller=cycle.MotionCycle.__new__(cycle.MotionCycle)
        result=controller.bootstrap('turkeyworld.collect_policy_motion','--cycle-index',3,
            '--output-dir',Path('a path with spaces'),'--device','cuda')
        self.assertEqual(result,[sys.executable,'-X','utf8','-u','-m','turkeyworld.bootstrap',
            '--module','turkeyworld.collect_policy_motion','--cycle-index','3',
            '--output-dir',str(Path('a path with spaces')),'--device','cuda'])
        self.assertFalse(any('training_runs/diagnostic_tools'in value for value in result))


class PortableBootstrapTests(unittest.TestCase):
    def test_argument_boundary_preserves_all_target_device_and_thread_flags(self):
        options,rest=bootstrap.arguments(['--device','cpu','--threads','3','--module','target.module',
            '--device','cuda','--threads','17','--value','a path with spaces'])
        self.assertEqual((options.device,options.threads,options.module),('cpu',3,'target.module'))
        self.assertEqual(rest,['--device','cuda','--threads','17','--value','a path with spaces'])
        for args in(['--threads','0','--module','target.module'],['--module'],['--module','']):
            with self.subTest(args=args),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                bootstrap.arguments(args)

    def test_torch_initialization_precedes_target_import_and_preserves_argv(self):
        calls=[]
        fake=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,
            init=lambda:calls.append('cuda')),set_num_threads=lambda n:calls.append(('threads',n)))
        original_argv=sys.argv
        def target(name,**kwargs):
            self.assertEqual(calls,['cuda',('threads',2)])
            self.assertEqual((os.environ['OMP_NUM_THREADS'],os.environ['MKL_NUM_THREADS']),('2','2'))
            self.assertEqual(sys.argv,['target.module','--device','cuda'])
            calls.append('target');return {'completed':True}
        with patch.dict(sys.modules,{'torch':fake}),patch.dict(os.environ,{},clear=False),\
                patch.object(bootstrap.runpy,'run_module',side_effect=target)as module,\
                patch.object(bootstrap.faulthandler,'enable'),\
                patch.object(bootstrap.faulthandler,'dump_traceback_later'),\
                patch.object(bootstrap.faulthandler,'cancel_dump_traceback_later'),redirect_stdout(io.StringIO()):
            result=bootstrap.main(['--module','target.module','--device','cuda'])
        module.assert_called_once_with('target.module',run_name='__main__',alter_sys=True)
        self.assertEqual(result,{'completed':True});self.assertIs(sys.argv,original_argv)

    def test_cpu_and_no_cuda_auto_do_not_touch_cuda_init(self):
        for mode,available in(('cpu',True),('auto',False)):
            fake=SimpleNamespace(cuda=SimpleNamespace(is_available=Mock(return_value=available),init=Mock()),
                set_num_threads=Mock())
            with self.subTest(mode=mode),patch.dict(sys.modules,{'torch':fake}),\
                    patch.dict(os.environ,{},clear=False):
                result=bootstrap.initialize_torch(device=mode,threads=2)
                self.assertEqual(result,{'cuda_initialized':False,'cpu_threads':2})
                fake.cuda.init.assert_not_called();fake.set_num_threads.assert_called_once_with(2)
                if mode=='cpu':fake.cuda.is_available.assert_not_called()

    def test_script_mode_runs_after_initialization_and_restores_paths_on_failure(self):
        with TemporaryDirectory()as directory:
            path=Path(directory)/'target script.py';path.write_text('raise RuntimeError("target")')
            argv,path_before=sys.argv,sys.path[:]
            def target(name,**kwargs):
                self.assertEqual(name,str(path.resolve()));self.assertEqual(sys.argv,[str(path.resolve()),'--x','1'])
                self.assertEqual(sys.path[0],str(path.parent.resolve()))
                raise RuntimeError('target')
            with patch.object(bootstrap,'initialize_torch',return_value={'cuda_initialized':False}),\
                    patch.object(bootstrap.runpy,'run_path',side_effect=target),\
                    patch.object(bootstrap.faulthandler,'enable'),\
                    patch.object(bootstrap.faulthandler,'dump_traceback_later'),\
                    patch.object(bootstrap.faulthandler,'cancel_dump_traceback_later'),redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'target'):
                    bootstrap.main(['--device','cpu','--script',str(path),'--x','1'])
            self.assertIs(sys.argv,argv);self.assertEqual(sys.path,path_before)

    def test_scheduler_and_bootstrap_help_import_without_torch_or_training_files(self):
        # Foreground CPU-only subprocess: validates a clean interpreter, not a
        # worker or training stage. --help must exit before torch/CUDA import.
        code='import sys; import turkeyworld.dreamer_motion_cycle; import turkeyworld.bootstrap; assert "torch" not in sys.modules'
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        result=subprocess.run([sys.executable,'-m','turkeyworld.bootstrap','--help'],
            capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--module',result.stdout);self.assertNotIn('[bootstrap]',result.stdout)


if __name__=='__main__':unittest.main()
