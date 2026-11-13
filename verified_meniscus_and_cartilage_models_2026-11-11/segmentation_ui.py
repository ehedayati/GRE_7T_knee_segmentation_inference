#!/usr/bin/env python3
"""Browser controls for the combined segmentation pipeline (Python standard library).

Run in the autoSeg environment on the server, then use the printed browser URL.
The web server binds to localhost; use SSH port forwarding from your workstation.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit

HERE = Path(__file__).resolve().parent
OUTPUT_NAMES = ['cartilage_only.nii.gz', 'meniscus_only.nii.gz',
                'merged_cartilage_meniscus.nii.gz', 'merged_cartilage_meniscus_merge.json']
ECHO_NAME = 'merged_cartilage_meniscus_all_echoes.nii.gz'
DICOM_NAMES = ['converted_input/gre.nii.gz', 'converted_input/gre.json']


class Panel:
    def __init__(self, start_dir=HERE.parent, pipeline=HERE/'run_combined_segmentation.sh', bash=None,
                 cartilage_dir=HERE.parent/'weights'/'cartilage',
                 meniscus_dir=HERE.parent/'weights'/'joint',
                 converter=HERE/'convert_gre_dicom.py'):
        self.start_dir = Path(start_dir).expanduser().resolve()
        self.pipeline = Path(pipeline).resolve()
        self.converter = Path(converter).resolve()
        self.bash = bash or shutil.which('bash')
        self.model_dirs = {'cartilage':Path(cartilage_dir).resolve(), 'meniscus':Path(meniscus_dir).resolve()}
        self.default_models = {
            'cartilage':self.model_dirs['cartilage']/'top3_unet2_4echo.pt',
            'meniscus':self.model_dirs['meniscus']/'joint_unet_4echo_five_losses.pt'}
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.thread = None
        self.process = None
        self.cancelled = False
        self.gpu_lock = threading.Lock()
        self.gpu_snapshot = None
        self.gpu_checked = 0.0
        self.logs = deque(maxlen=1500)
        self.state = {'state':'idle', 'stage':0, 'total':5, 'message':'Ready',
                      'output_dir':'', 'outputs':[], 'exit_code':None, 'settings':None}

    def status(self):
        with self.lock:
            return {**self.state, 'logs':list(self.logs), 'start_dir':str(self.start_dir),
                    'python':sys.executable}

    @staticmethod
    def model_info(path, role):
        """Validate metadata without importing a network or loading its weights."""
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f'{role.capitalize()} checkpoint not found: {path}')
        manifest_path = path.with_name('manifest.json')
        if not manifest_path.is_file():
            raise ValueError(f'Checkpoint needs a matching manifest.json beside it: {path}')
        with manifest_path.open(encoding='utf-8') as stream:
            manifest = json.load(stream)
        if manifest.get('schema_version') != 1:
            raise ValueError(f'Unsupported model manifest schema: {manifest_path}')
        matches = [entry for entry in manifest.get('models',[]) if entry.get('checkpoint')==path.name]
        if len(matches) != 1:
            raise ValueError(f'Expected one manifest entry for {path.name}')
        entry = matches[0]
        cfg = entry.get('inference',{})
        expected = 3 if role=='cartilage' else 5
        if cfg.get('out_channel') != expected or cfg.get('Network') not in ('UNet','UNet2'):
            raise ValueError(f'{role.capitalize()} requires a supported UNet/UNet2 with {expected} output classes')
        echoes = cfg.get('specific_echos',[])
        if (not isinstance(echoes,list) or not echoes or
            any(type(index) is not int or index<0 for index in echoes) or
            cfg.get('num_echos') != len(echoes)):
            raise ValueError(f'Invalid echo metadata for {path.name}')
        required = ('Unet_res_no','normalize','z_score','crop_size','window_size')
        if any(key not in cfg for key in required) or not re.fullmatch('[0-9a-fA-F]{64}',str(entry.get('sha256',''))):
            raise ValueError(f'Incomplete inference settings or checksum for {path.name}')
        return {'path':str(path), 'name':entry.get('name',path.stem),
                'network':cfg['Network'], 'echoes':echoes,
                'classes':expected, 'sha256':entry['sha256'], 'crop_size':cfg['crop_size']}

    def models(self):
        result = {'defaults':{key:str(path) for key,path in self.default_models.items()},
                  'cartilage':[], 'meniscus':[], 'errors':[]}
        for role,folder in self.model_dirs.items():
            try:
                with (folder/'manifest.json').open(encoding='utf-8') as stream:
                    manifest = json.load(stream)
                for entry in manifest.get('models',[]):
                    try:
                        path = (folder/entry['checkpoint']).resolve()
                        if path.parent != folder:
                            raise ValueError('Catalog checkpoints must be inside their model directory')
                        info = self.model_info(path,role)
                        info['default'] = path==self.default_models[role]
                        result[role].append(info)
                    except (OSError,ValueError,KeyError,TypeError) as error:
                        result['errors'].append(str(error))
            except (OSError,ValueError,TypeError) as error:
                result['errors'].append(f'{role.capitalize()} catalog: {error}')
        return result

    def gpu_availability(self, fresh=False):
        # Independent lock: status polling/stopping stays responsive during queries.
        with self.gpu_lock:
            if not fresh and self.gpu_snapshot is not None and time.monotonic()-self.gpu_checked < 10:
                return self.gpu_snapshot
            try:
                query = subprocess.run([sys.executable, '-B', str(HERE/'gpu_availability.py')],
                                       capture_output=True, text=True, encoding='utf-8',
                                       errors='replace', timeout=25, stdin=subprocess.DEVNULL)
                if query.returncode:
                    raise ValueError(query.stderr.strip() or 'GPU query failed')
                result = json.loads(query.stdout)
                if not isinstance(result.get('devices'),list):
                    raise ValueError('Invalid GPU query response')
            except (OSError,ValueError,subprocess.TimeoutExpired) as error:
                result = {'devices':[], 'cuda_available':None,
                          'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                          'checked_at':datetime.now().isoformat(timespec='seconds'),
                          'error':str(error)}
            self.gpu_snapshot = result
            self.gpu_checked = time.monotonic()
            return result

    def resolve_device(self, requested):
        if requested == 'cpu':
            return 'cpu', 'CPU selected'
        snapshot = self.gpu_availability(fresh=True)
        if snapshot.get('error'):
            raise ValueError('Cannot check GPU availability: '+snapshot['error'])
        devices = snapshot['devices']
        readable = [gpu for gpu in devices if not gpu.get('error') and
                    isinstance(gpu.get('free_bytes'),int) and gpu['free_bytes'] > 0]
        if requested == 'auto':
            if not snapshot['cuda_available'] and not devices:
                return 'cpu', 'No CUDA GPU visible to this process; auto selected CPU'
            if not readable:
                raise ValueError('No visible GPU has readable free memory. Refresh GPU availability or select CPU.')
            gpu = max(readable,key=lambda item:item['free_bytes'])
        else:
            device = 'cuda:0' if requested == 'cuda' else requested
            matches = [item for item in readable if item['device']==device]
            if not matches:
                raise ValueError(f'{device} is not visible/usable in this session. Refresh GPU availability.')
            gpu = matches[0]
        message = (f"Selected {gpu['device']} ({gpu['name']}), "
                   f"{gpu['free_bytes']/1024**3:.1f} GiB free of {gpu['total_bytes']/1024**3:.1f} GiB")
        return gpu['device'], message

    def browse(self, raw_path, mode):
        path = Path(raw_path or self.start_dir).expanduser().resolve()
        if not path.is_dir():
            raise ValueError('Choose an existing server directory')
        entries = []
        with os.scandir(path) as items:
            for item in items:
                if item.name.startswith('.'):
                    continue
                try:
                    directory = item.is_dir()
                    if directory or ((mode == 'file' and item.name.lower().endswith(('.nii','.nii.gz')) or
                                      mode == 'checkpoint' and item.name.lower().endswith(('.pt','.pth','.ckpt'))) and item.is_file()):
                        entries.append({'name':item.name, 'path':str(path/item.name), 'directory':directory})
                except OSError:
                    continue
        entries.sort(key=lambda item:(not item['directory'], item['name'].lower()))
        return {'path':str(path), 'parent':str(path.parent), 'entries':entries[:1500],
                'truncated':len(entries)>1500}

    def start(self, data):
        input_text = str(data.get('input','')).strip()
        split_output = 'output_parent' in data or 'output_name' in data
        if split_output:
            parent_text = str(data.get('output_parent','')).strip()
            name = str(data.get('output_name','')).strip()
            if not parent_text or not name:
                raise ValueError('Choose an output parent folder and enter a new results folder name')
            if name in ('.','..') or '/' in name or '\\' in name or any(ord(c)<32 for c in name):
                raise ValueError('Results folder name must be a single name without slashes, such as output_dir_name')
            parent = Path(parent_text).expanduser().resolve()
            if not parent.is_dir():
                raise ValueError('Output parent folder must be an existing server directory')
            output = parent/name
            if output.exists() or output.is_symlink():
                raise ValueError(f'Results folder already exists: {output}. Enter a new folder name.')
        else:
            # Compatibility with callers that submit a complete output path.
            output_text = str(data.get('output','')).strip()
            if not output_text:
                raise ValueError('Select an output directory')
            output = Path(output_text).expanduser().resolve()
        if not input_text:
            raise ValueError('Select an input MRI')
        source = Path(input_text).expanduser().resolve()
        input_type = str(data.get('input_type','nifti')).strip()
        if input_type not in ('nifti','dicom'):
            raise ValueError('Input type must be nifti or dicom')
        if input_type == 'dicom':
            if not source.is_dir():
                raise ValueError('GRE DICOM input must be an existing series folder on the server')
            if not self.converter.is_file():
                raise ValueError('Keep convert_gre_dicom.py beside segmentation_ui.py')
            converted_dir = output/'converted_input'
            if converted_dir.exists() or converted_dir.is_symlink():
                raise ValueError('Converted input directory already exists. Choose a new output folder.')
        elif not source.is_file() or not source.name.lower().endswith(('.nii','.nii.gz')):
            raise ValueError('Input must be an existing .nii or .nii.gz file on the server')
        if output.is_symlink() or (output.exists() and not output.is_dir()):
            raise ValueError('Output must be a directory')
        device = str(data.get('device','auto')).strip()
        if not re.fullmatch(r'auto|cpu|cuda(?::\d+)?', device):
            raise ValueError('Device must be auto, cpu, cuda, or cuda:N')
        try:
            radius = float(data.get('radius',1.0))
        except (TypeError,ValueError) as error:
            raise ValueError('Correction radius must be a positive number in mm') from error
        if not math.isfinite(radius) or not 0 < radius <= 100:
            raise ValueError('Correction radius must be greater than 0 and no more than 100 mm')
        echoes = data.get('all_echoes',True)
        if not isinstance(echoes,bool):
            raise ValueError('All-echo output must be true or false')
        if not self.bash or not self.pipeline.is_file():
            raise ValueError('Bash or run_combined_segmentation.sh is unavailable')
        names = (DICOM_NAMES if input_type == 'dicom' else []) + OUTPUT_NAMES + ([ECHO_NAME] if echoes else [])
        for name in names:
            path = output/name
            if path.exists() or path.is_symlink():
                raise ValueError(f'Output already exists: {path}. Choose a new output folder.')
        cartilage = self.model_info(data.get('cartilage_model') or self.default_models['cartilage'],'cartilage')
        meniscus = self.model_info(data.get('meniscus_model') or self.default_models['meniscus'],'meniscus')
        settings = {'cartilage_model':cartilage['path'], 'meniscus_model':meniscus['path'],
                    'input':str(source), 'input_type':input_type, 'output':str(output), 'device':device,
                    'required_echoes':max(cartilage['echoes']+meniscus['echoes'])+1,
                    'min_xy':2*max(cartilage['crop_size'],meniscus['crop_size']),
                    'radius':radius, 'all_echoes':echoes,
                    'output_parent':str(output.parent), 'output_name':output.name}
        with self.lock:
            if self.state['state'] in ('running','stopping'):
                raise ValueError('A scan is already running. Wait for it to finish.')
            self.cancelled = False
            self.logs.clear()
            self.state.update(state='running', stage=0, total=(5 if echoes else 4)+(input_type == 'dicom'),
                              message='Starting pipeline', output_dir=str(output), outputs=[],
                              exit_code=None, settings=settings,
                              started=datetime.now().isoformat(timespec='seconds'))
            self.thread = threading.Thread(target=self._run, args=(settings,names), daemon=True)
            self.thread.start()
        return self.status()

    def _run(self, settings, names):
        env = os.environ.copy()
        env['PYTHON_BIN'] = sys.executable
        env['ALL_ECHOES'] = '1' if settings['all_echoes'] else '0'
        env['CARTILAGE_MODEL_PATH'] = settings['cartilage_model']
        env['MENISCUS_MODEL_PATH'] = settings['meniscus_model']
        code = None
        error_message = None
        try:
            with self.lock:
                self.logs.append('Python environment: '+sys.executable)
                self.logs.append('Cartilage checkpoint: '+settings['cartilage_model'])
                self.logs.append('Meniscus checkpoint: '+settings['meniscus_model'])
                if self.cancelled:
                    return
                if settings['device'] != 'cpu':
                    self.state['message'] = 'Checking visible GPUs before starting'
            device, selection_message = self.resolve_device(settings['device'])
            with self.lock:
                settings['resolved_device'] = device
                self.logs.append(selection_message)
                if self.cancelled:
                    return
            source = settings['input']
            offset = int(settings['input_type'] == 'dicom')
            if offset:
                source = str(Path(settings['output'])/DICOM_NAMES[0])
                with self.lock:
                    self.state.update(stage=1, message='Convert GRE DICOM with MRIForge')
                    self.logs.append(f"[1/{self.state['total']}] Convert GRE DICOM with MRIForge")
                command = [sys.executable, '-u', str(self.converter), '--input', settings['input'],
                           '--output', source, '--required-echoes', str(settings['required_echoes']),
                           '--min-xy', str(settings['min_xy'])]
                conversion_code = self._execute(command, env, stage_offset=None)
                with self.lock:
                    if self.cancelled:
                        return
                if conversion_code != 0:
                    code = conversion_code
                    raise RuntimeError(f'DICOM conversion failed (exit {code}). See the log. Segmentation was not started.')
                if not all((Path(settings['output'])/name).is_file() for name in DICOM_NAMES):
                    raise RuntimeError('DICOM conversion did not create the NIfTI and JSON sidecar. Segmentation was not started.')
            with self.lock:
                settings['inference_input'] = source
                if self.cancelled:
                    return
            command = [self.bash, str(self.pipeline), source, settings['output'],
                       device, format(settings['radius'],'.10f')]
            code = self._execute(command, env, stage_offset=offset)
        except Exception as error:
            error_message = str(error)
            with self.lock:
                self.logs.append('ERROR: '+error_message)
        finally:
            with self.lock:
                state = 'cancelled' if self.cancelled else ('completed' if code==0 else 'failed')
                self.state.update(state=state, exit_code=code,
                                  message={'cancelled':'Stopped by user', 'completed':'Completed',
                                           'failed':error_message or f'Pipeline failed (exit {code}). See the log.'}[state],
                                  outputs=[str(Path(settings['output'])/name) for name in names
                                           if (Path(settings['output'])/name).is_file()])
                self.process = None

    def _execute(self, command, env, stage_offset=0):
        """Run a cancellable stage using the UI interpreter and shared log."""
        with self.lock:
            if self.cancelled:
                return None
        process = subprocess.Popen(command, cwd=self.pipeline.parent, env=env,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                   errors='replace', bufsize=1, start_new_session=(os.name=='posix'))
        with self.lock:
            self.process = process
            cancelled = self.cancelled
        if cancelled:
            self._terminate(process)
        try:
            with process.stdout:
                for line in process.stdout:
                    line = line.rstrip()
                    stage = re.search(r'\[(\d+)/(\d+)\]\s*(.*)',line) if stage_offset is not None else None
                    with self.lock:
                        if stage:
                            number, total = int(stage[1])+stage_offset, int(stage[2])+stage_offset
                            line = line[:stage.start()]+f'[{number}/{total}] '+stage[3]
                            self.state.update(stage=number, total=total, message=stage[3])
                        self.logs.append(line)
            return process.wait()
        finally:
            with self.lock:
                if self.process is process:
                    self.process = None

    @staticmethod
    def _terminate(process):
        if process.poll() is None:
            try:
                if os.name=='posix':
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            except ProcessLookupError:
                pass

    def stop(self):
        with self.lock:
            if self.state['state'] not in ('running','stopping'):
                return
            self.cancelled = True
            self.state.update(state='stopping', message='Stopping the current run')
            process = self.process
        if process is not None:
            self._terminate(process)


def make_handler(panel):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Keep the browser token out of request logs.

        def respond(self, value, status=200):
            body = json.dumps(value,allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            token = self.headers.get('Authorization','').removeprefix('Bearer ')
            return secrets.compare_digest(token,panel.token)

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == '/':
                body = (HERE/'segmentation_ui.html').read_bytes()
                self.send_response(200)
                self.send_header('Content-Type','text/html; charset=utf-8')
                self.send_header('Content-Length',str(len(body)))
                self.send_header('Cache-Control','no-store')
                self.send_header('Referrer-Policy','no-referrer')
                self.end_headers()
                self.wfile.write(body)
                return
            if not self.authorized():
                self.respond({'error':'Open the full URL printed in the server terminal, including its token.'},403)
                return
            try:
                if url.path == '/api/status':
                    self.respond(panel.status())
                elif url.path == '/api/models':
                    self.respond(panel.models())
                elif url.path == '/api/gpus':
                    self.respond(panel.gpu_availability(fresh=True))
                elif url.path == '/api/browse':
                    query = parse_qs(url.query)
                    self.respond(panel.browse(query.get('path',[''])[0],query.get('mode',['file'])[0]))
                else:
                    self.respond({'error':'Not found'},404)
            except (ValueError,OSError) as error:
                self.respond({'error':str(error)},400)

        def do_POST(self):
            if not self.authorized():
                self.respond({'error':'Invalid session token'},403)
                return
            try:
                size = int(self.headers.get('Content-Length','0'))
                if not 0 <= size <= 16384:
                    raise ValueError('Request too large')
                data = json.loads(self.rfile.read(size) or b'{}')
                if not isinstance(data,dict):
                    raise ValueError('Expected a JSON object')
                if self.path == '/api/run':
                    self.respond(panel.start(data))
                elif self.path == '/api/stop':
                    panel.stop()
                    self.respond(panel.status())
                else:
                    self.respond({'error':'Not found'},404)
            except (ValueError,OSError,TypeError) as error:
                self.respond({'error':str(error)},400)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8501)
    parser.add_argument('--start-dir',type=Path,default=HERE.parent,help='Initial server file-browser directory')
    args = parser.parse_args()
    panel = Panel(args.start_dir)
    if not (HERE/'segmentation_ui.html').is_file():
        parser.error('Keep segmentation_ui.html beside segmentation_ui.py')
    if not panel.start_dir.is_dir():
        parser.error('--start-dir must be an existing directory')
    server = ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(panel))
    print(f'Open: http://127.0.0.1:{server.server_port}/?token={panel.token}',flush=True)
    print(f'Python: {sys.executable}',flush=True)
    print('For a remote server, forward this port over SSH. Ctrl+C stops this UI and its active run.',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        panel.stop()
        if panel.thread is not None:
            panel.thread.join(timeout=10)
        server.server_close()


if __name__=='__main__':
    main()
