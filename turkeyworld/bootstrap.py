"""Initialize PyTorch/CUDA before importing a training or playback entry point.

Example: python3 -m turkeyworld.bootstrap --module turkeyworld.collect_policy_motion ...
Bootstrap options precede --module/--script; every subsequent argument belongs
to the target, including its own --device and --threads options. This module
does not initialize CUDA, import torch or start a worker when imported.
"""
from __future__ import annotations

import argparse
import faulthandler
import os
from pathlib import Path
import runpy
import sys


def arguments(argv=None):
    values=list(sys.argv[1:]if argv is None else argv)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('auto','cuda','cpu'),default='auto',
                        help='Bootstrap initialization only; does not replace target device arguments.')
    parser.add_argument('--threads',type=int,default=2)
    target=parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--module')
    target.add_argument('--script',type=Path)
    # Do not consume a target's --device/--threads or reinterpret its flags.
    boundary=next((index for index,item in enumerate(values)if item in('--module','--script')),None)
    prefix=values if boundary is None else values[:boundary+2]
    remaining=[]if boundary is None else values[boundary+2:]
    options=parser.parse_args(prefix)
    if options.threads<1:parser.error('--threads must be positive')
    if options.module is not None and not options.module.strip():parser.error('--module must be nonempty')
    if options.script is not None:
        options.script=options.script.resolve()
        if not options.script.is_file():parser.error(f'Script not found: {options.script}')
    return options,remaining


def initialize_torch(*,device,threads):
    # Before importing torch and its OpenMP/MKL libraries, on either platform.
    os.environ['OMP_NUM_THREADS']=str(threads)
    os.environ['MKL_NUM_THREADS']=str(threads)
    import torch
    initialize_cuda=device=='cuda'or(device=='auto'and torch.cuda.is_available())
    if initialize_cuda:torch.cuda.init()
    torch.set_num_threads(threads)
    return dict(cuda_initialized=initialize_cuda,cpu_threads=threads)


def main(argv=None):
    options,remaining=arguments(argv)
    faulthandler.enable()
    faulthandler.dump_traceback_later(45,repeat=False)
    try:
        initialized=initialize_torch(device=options.device,threads=options.threads)
    finally:
        faulthandler.cancel_dump_traceback_later()
    print(f'[bootstrap] CUDA initialized={initialized["cuda_initialized"]}; CPU threads={options.threads}',flush=True)
    target=options.module if options.module is not None else str(options.script)
    original_argv,original_path=sys.argv,sys.path[:]
    sys.argv=[target,*remaining]
    try:
        if options.module is not None:
            return runpy.run_module(target,run_name='__main__',alter_sys=True)
        sys.path.insert(0,str(options.script.parent))
        return runpy.run_path(target,run_name='__main__')
    finally:
        sys.argv=original_argv
        sys.path[:]=original_path


if __name__=='__main__':main()
