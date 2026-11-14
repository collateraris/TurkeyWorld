"""Local read-only policy monitor; independent of the running dynamics page."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import hashlib
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
PAGE_ROOT=ROOT/'policy_dashboard'


@lru_cache(maxsize=64)
def _sha_cached(path,size,mtime_ns):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _sha(path):
    path=Path(path);stat=path.stat();return _sha_cached(str(path),stat.st_size,stat.st_mtime_ns)


def verified_validation(root,receipt):
    """Verify pinned saved evaluation artifacts without deserializing weights."""
    def inside(name):
        path=Path(receipt[name]);path=path.resolve()if path.is_absolute()else(root/path).resolve()
        if root not in path.parents:raise ValueError('Policy validation artifact escapes its run directory')
        return path
    report_path=inside('report_file');policy_path=inside('policy_checkpoint_file');world_path=inside('world_checkpoint_file')
    if(_sha(report_path)!=receipt['report_sha256']or _sha(policy_path)!=receipt['policy_checkpoint_sha256']
            or _sha(world_path)!=receipt['world_sha256']):raise ValueError('Saved policy validation file SHA changed')
    report=json.loads(report_path.read_text(encoding='utf-8'))
    expected=dict(panel_fingerprint=receipt['panel_fingerprint'],data_identity_sha256=receipt['validation_data_identity_sha256'],step=receipt['policy_step'])
    if(any(report.get(key)!=value for key,value in expected.items())
            or report.get('policy_identity',{}).get('checkpoint_sha256')!=receipt['policy_checkpoint_sha256']
            or report.get('policy_identity',{}).get('plan_sha256')!=receipt['plan_sha256']
            or report.get('world_identity',{}).get('checkpoint_sha256')!=receipt['world_sha256']
            or report.get('world_identity',{}).get('step')!=receipt['world_step']):
        raise ValueError('Saved evaluation receipt disagrees with policy/world/panel identities')
    return report


def monitor_payload(run_dir,*,history_limit=2048):
    root=Path(run_dir).resolve();path=root/'status.json'
    if not path.is_file():return dict(status='waiting',message='Policy training has not published status.json',history=[])
    value=json.loads(path.read_text(encoding='utf-8'));history=[];log=root/'history.jsonl'
    if log.is_file():
        for line in log.read_text(encoding='utf-8').splitlines()[-history_limit:]:
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            if isinstance(row,dict):history.append(row)
    value['history']=history;validation=value.get('validation',{})
    if isinstance(value.get('validation_receipt'),dict):
        try:value['validation_report']=verified_validation(root,value['validation_receipt']);value['validation_receipt_verified']=True
        except(ValueError,KeyError,OSError,json.JSONDecodeError)as error:
            value['validation_receipt_verified']=False;value['validation_receipt_error']=str(error)
        return value
    if isinstance(validation,dict)and validation.get('file'):
        report=Path(validation['file']);report=report.resolve()if report.is_absolute()else(root/report).resolve()
        if root not in report.parents:raise ValueError('Policy validation file must remain in the explicit run directory')
        if report.is_file():value['validation_report']=json.loads(report.read_text(encoding='utf-8'))
    return value


def handler(run_dir):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route=urlsplit(self.path).path
            try:
                if route=='/api/status':body=json.dumps(monitor_payload(run_dir),ensure_ascii=False,allow_nan=False).encode('utf-8');mime='application/json; charset=utf-8'
                elif route in('/','/index.html','/view.mjs'):
                    file=PAGE_ROOT/('view.mjs'if route=='/view.mjs'else'index.html');body=file.read_bytes()
                    mime='text/javascript; charset=utf-8'if file.suffix=='.mjs'else'text/html; charset=utf-8'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store')
                self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            except(FileNotFoundError,json.JSONDecodeError,ValueError)as error:
                body=json.dumps({'error':str(error)}).encode();self.send_response(503);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    return Handler


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--port',type=int,default=8787)
    a=p.parse_args(argv)
    if not 1024<=a.port<=65535:p.error('Use an explicit unprivileged local port')
    ThreadingHTTPServer(('127.0.0.1',a.port),handler(a.run_dir)).serve_forever()


if __name__=='__main__':main()
