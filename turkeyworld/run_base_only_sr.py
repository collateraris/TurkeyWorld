"""Prepare immutable Base inputs, then train the independent Base-only SR branch."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from .training_pipeline import atomic_json,now
from .small_enemy_data import digest
from .power_guard import KeepAwake


def run(a):
    root=a.run_dir.resolve();assets=root/'assets';assets.mkdir(parents=True,exist_ok=True)
    for key in ('world','renderer','sr_source'):
        src=getattr(a,key).resolve();dst=assets/('source.pt' if key=='sr_source' else key+'.pt')
        if dst.exists():
            if digest(src)!=digest(dst):raise ValueError('Pinned Base branch asset changed')
        else:os.link(src,dst)
    command=[sys.executable,'-m','turkeyworld.cache_base_only_sr','--world',str(assets/'world.pt'),
        '--renderer',str(assets/'renderer.pt'),'--source-root',str(a.source_pairs.resolve()),'--output',str(root/'data/paired')]
    atomic_json(root/'launch_plan.json',dict(protocol='base_only_sr_pipeline_v1',flow_matching=False,codec=False,
        world_sha256=digest(assets/'world.pt'),renderer_sha256=digest(assets/'renderer.pt'),sr_source_sha256=digest(assets/'source.pt'),
        source_pairs=str(a.source_pairs.resolve()),target_updates=a.steps,created_at=now()))
    with (root/'cache.stdout.log').open('ab') as out,(root/'cache.stderr.log').open('ab') as err:
        child=subprocess.Popen(command,stdout=out,stderr=err)
        while child.poll() is None:
            path=root/'data/paired/manifest.json';v=json.loads(path.read_text()) if path.exists() else {}
            atomic_json(root/'status.json',dict(stage='data',status='running',run_id=root.name,
                model='Base-only input preparation (RF/VQGAN disabled)',current=v.get('current',0),total=v.get('total',9216),updated_at=now()))
            if (root/'stop_training').exists():child.terminate();child.wait();return
            time.sleep(2)
        if child.returncode:raise RuntimeError('Base cache failed; inspect cache.stderr.log')
    subprocess.run([sys.executable,'-m','turkeyworld.train_rrdb_material_sr','--run-dir',str(root),
        '--data-root',str(root/'data/paired'),'--base-only','--steps',str(a.steps)],check=True)


def main():
    p=argparse.ArgumentParser()
    for key in ('run-dir','world','renderer','sr-source','source-pairs'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--steps',type=int,default=2000);a=p.parse_args()
    try:
        with KeepAwake():run(a)
    except BaseException as error:
        atomic_json(a.run_dir/'status.json',dict(stage='sr',status='failed',error=str(error),updated_at=now()));raise

if __name__=='__main__':main()
