"""Headless periodic V-JEPA held-out previews; never opens a game window."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from .process_runtime import pid_file_alive,write_pid_file
from .training_pipeline import atomic_json,now


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();root=a.run_dir.resolve()
    if pid_file_alive(root/'preview.pid'):raise ValueError('V-JEPA preview worker already running')
    write_pid_file(root/'preview.pid');(root/'preview_assets').mkdir(exist_ok=True)
    while not (root.parent/'stop.request').exists():
        try:
            status=json.loads((root/'status.json').read_text());preview=json.loads((root/'preview.json').read_text()) if (root/'preview.json').exists() else {}
            step=status.get('current',0)
            if step and (not preview or step-preview.get('step',0)>=10 or status.get('status')=='world_round_complete' and step!=preview.get('step')):
                snapshot=root/'preview_assets/world.pt'
                temp=root/'preview_assets/world.tmp.pt'
                if temp.exists():temp.unlink()
                os.link(root/'checkpoints/vjepa_world.pt',temp);temp.replace(snapshot)
                atomic_json(root/'preview_progress.json',dict(status='running',requested_step=step,updated_at=now()))
                with (root/'preview.stdout.log').open('ab') as out,(root/'preview.stderr.log').open('ab') as err:
                    child=subprocess.Popen([sys.executable,'-m','turkeyworld.validate_vjepa_sr','--world',str(snapshot),
                        '--output',str(root/'preview_assets/validation.json'),'--preview-dir',str(root)],stdout=out,stderr=err)
                    while child.poll() is None:
                        if (root.parent/'stop.request').exists():child.terminate();child.wait();return
                        time.sleep(1)
                atomic_json(root/'preview_progress.json',dict(status='complete' if child.returncode==0 else 'failed',updated_at=now()))
        except (OSError,ValueError) as error:
            atomic_json(root/'preview_progress.json',dict(status='retrying',error=str(error),updated_at=now()))
        time.sleep(2)

if __name__=='__main__':main()
