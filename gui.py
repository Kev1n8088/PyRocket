#####################################################################
#                           PyRocket								#
# 2D Regenertive Cooling Simulation for Bipropellant Rocket Engines #
#                                                                   #
# Creator:  Joanthan Neeser                                         #
# Date:     15.12.2022                                              #
# Version:  2.3  													#
# License:	GNU GENERAL PUBLIC LICENSE V3							#
#####################################################################

# Browser interface for single runs, batch runs and helix sweeps. Starts a local web server and runs run.py and sweep.py
# in the background, so results are the same as from the command line.
#
# Usage:
#   python3 gui.py                  then open http://localhost:8765 in a browser
#   python3 gui.py --port 9000
#
# Only listens on this computer (127.0.0.1).

import argparse
import csv
import glob
import itertools
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import Units

REPO    = os.path.dirname(os.path.abspath(__file__))
STATIC  = os.path.join(REPO, 'webui')
LOGS    = os.path.join(tempfile.gettempdir(), 'pyrocket_gui')
SECTION = re.compile(r'SOLVING Section Number\s+(\d+)\s*/\s*(\d+)')
STAGE   = re.compile(r'(Grid|Refinement (\d+)/(\d+)): designs evaluated (\d+) / (\d+)')

# key statistics shown for a case: summary key, label, unit, conversion from SI
TEMP = Units.temperature_unit()
STATS = [
    ('T_wall_max',        'Max inner wall temp',        TEMP,    Units.temperature),
    ('T_wall_median',     'Median inner wall temp',     TEMP,    Units.temperature),
    ('T_wall_throat',     'Inner wall temp at throat',  TEMP,    Units.temperature),
    ('T_coolant_out',     'Coolant outlet temp',        TEMP,    Units.temperature),
    ('dp_coolant',        'Coolant pressure drop',      'bar',   lambda v: v / 1e5),
    ('dp_injector',       'Coolant outlet P minus Pc',  'bar',   lambda v: v / 1e5),
    ('q_max',             'Peak heat flux',             'MW/m²', lambda v: v / 1e6),
    ('Q_total',           'Total heat load',            'kW',    lambda v: v / 1e3),
    ('boil_margin_wall',  'Coolant wall below boiling', 'K',     lambda v: v),
    ('stress_margin_min', 'Minimum wall stress margin', '-',     lambda v: v),
    ('sigma_vm_at_min_margin', 'Von Mises stress there', 'MPa', lambda v: v / 1e6),
    ('yield_at_min_margin', 'Yield strength there',  'MPa',   lambda v: v / 1e6),
    ('v_coolant_max',     'Max coolant velocity',       'm/s',   lambda v: v),
    ('w_channel_min',     'Narrowest channel',          'mm',    lambda v: v * 1e3),
    ('w_rib_min',         'Narrowest rib',              'mm',    lambda v: v * 1e3),
    ('alpha_max_deg',     'Max helix angle',            'deg',   lambda v: v),
    ('dp_turnaround',     'Two-pass turnaround loss',   'bar',   lambda v: v / 1e5),
    ('T_coolant_turnaround', 'Coolant temp at turnaround', TEMP, Units.temperature),
    ('runtime_s',         'Run time',                   's',     lambda v: v),
]

jobs = {}
jobs_lock = threading.Lock()
job_ids = itertools.count(1)
allowed_roots = {os.path.realpath(REPO)}


def rel(path):
    # path relative to the repository where possible, for display
    path = os.path.realpath(path)
    return os.path.relpath(path, REPO) if path.startswith(os.path.realpath(REPO) + os.sep) else path


def resolve(path):
    # absolute path of a user supplied path, relative paths are relative to the repository
    return os.path.realpath(path if os.path.isabs(path) else os.path.join(REPO, path))


def allowed(path):
    path = os.path.realpath(path)
    return any(path == root or path.startswith(root + os.sep) for root in allowed_roots)


def list_configs():
    configs = sorted(glob.glob(os.path.join(REPO, 'configs', '*.py')))
    configs = [c for c in configs if not os.path.basename(c).startswith('_')]
    root = [c for c in sorted(glob.glob(os.path.join(REPO, 'config*.py')))]
    return [rel(c) for c in root + configs]


def case_name(config_path):
    return os.path.splitext(os.path.basename(config_path))[0]


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def section_progress(log_path):
    # last "SOLVING Section Number k / N" in a case log
    try:
        with open(log_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 8192))
            text = f.read().decode(errors='replace')
    except OSError:
        return None
    found = SECTION.findall(text)
    return (int(found[-1][0]), int(found[-1][1])) if found else None


def tail(path, lines=80):
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 20000))
            text = f.read().decode(errors='replace')
    except OSError:
        return ''
    text = '\n'.join(l for l in text.splitlines() if not l.startswith('USER_HOME_DIR'))
    return '\n'.join(text.splitlines()[-lines:])


def format_stats(summary):
    # key statistics of a case in display units
    rows = []
    for key, label, unit, conv in STATS:
        value = summary.get(key)
        if isinstance(value, (int, float)):
            rows.append({'key': key, 'label': label, 'unit': unit, 'value': conv(value)})
    return rows


def case_progress(folder, started):
    # state of a case folder written by run_case: done / failed (finished after the job started), running with section progress, or waiting
    summary = read_json(os.path.join(folder, 'summary.json'))
    if summary is not None and os.path.getmtime(os.path.join(folder, 'summary.json')) >= started - 1:
        return {'state': 'failed' if summary.get('status') != 'ok' else 'done', 'fraction': 1.0,
                'error': summary.get('error'), 'T_wall_max': summary.get('T_wall_max')}
    log = os.path.join(folder, 'log.txt')
    if os.path.isfile(log) and os.path.getmtime(log) >= started - 1:
        p = section_progress(log)
        return {'state': 'running', 'fraction': p[0] / p[1] if p else 0.0, 'sections': p}
    return {'state': 'waiting', 'fraction': 0.0}


class Job:
    def __init__(self, kind, title, cmd, out, cases=None, folder=None, params=None):
        self.id      = next(job_ids)
        self.kind    = kind            # 'run' or 'sweep'
        self.title   = title
        self.cmd     = cmd
        self.out     = out             # output folder of run.py / sweep.py
        self.cases   = cases or []     # case names of a run
        self.folder  = folder          # sweep folder
        self.params  = params or {}
        self.started = time.time()
        self.ended   = None
        self.stopped = False
        os.makedirs(LOGS, exist_ok=True)
        self.log = os.path.join(LOGS, 'job_%d_%d.log' % (os.getpid(), self.id))
        with open(self.log, 'w') as log:
            self.proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                         env=dict(os.environ, PYTHONUNBUFFERED='1'))
        allowed_roots.add(os.path.realpath(out))

    def status(self):
        code = self.proc.poll()
        if code is None:
            return 'running'
        if self.ended is None:
            self.ended = time.time()
        if self.stopped:
            return 'stopped'
        return 'finished' if code == 0 else 'failed'

    def stop(self):
        if self.proc.poll() is None:
            self.stopped = True
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def progress(self):
        if self.kind == 'optimize':
            return optimize_progress(self)
        if self.kind == 'run':
            cases = [dict(name=c, **case_progress(os.path.join(self.out, c), self.started)) for c in self.cases]
            fraction = sum(c['fraction'] for c in cases) / max(len(cases), 1)
            return {'cases': cases, 'fraction': fraction}
        points = []
        for folder in sorted(glob.glob(os.path.join(self.folder, self.params['param'] + '_*'))):
            if os.path.isdir(folder):
                p = case_progress(folder, self.started)
                if p['state'] == 'waiting' and os.path.isfile(os.path.join(folder, 'summary.json')):
                    # finished in an earlier sweep with the same config, reused by sweep.py
                    p = {'state': 'reused', 'fraction': 1.0, 'T_wall_max': (read_json(os.path.join(folder, 'summary.json')) or {}).get('T_wall_max')}
                if p['state'] != 'waiting':
                    points.append(dict(name=os.path.basename(folder), **p))
        points.sort(key=lambda p: float(p['name'].split('_')[-1]))
        done = sum(1 for p in points if p['state'] in ('done', 'failed', 'reused'))
        expected = self.params['points'] + 2 * self.params['refine']
        return {'cases': points, 'fraction': min(done / max(expected, 1), 0.99), 'done': done}

    def summary(self, detail=False):
        status = self.status()
        progress = self.progress()
        if status != 'running':
            progress['fraction'] = 1.0
            for case in progress['cases']:
                if case['state'] in ('running', 'waiting'):
                    case['state'] = 'stopped' if status == 'stopped' else 'failed'
                    case.pop('sections', None)
        info = {'id': self.id, 'kind': self.kind, 'title': self.title, 'status': status, 'started': self.started,
                'elapsed': (self.ended or time.time()) - self.started, 'out': rel(self.out), 'progress': progress,
                'command': ' '.join(os.path.basename(c) if i == 0 else c for i, c in enumerate(self.cmd))}
        if detail:
            info['log'] = tail(self.log)
            if status != 'running' and self.kind == 'optimize':
                info['result'] = optimize_result(self.folder, self.log)
            elif status != 'running':
                info['result'] = (run_result(self.out, self.cases) if self.kind == 'run' else sweep_result(self.folder))
                if self.kind == 'sweep':
                    # outcome lines printed by sweep.py, and the value of the best point
                    lines = [l for l in tail(self.log, 400).splitlines() if l.startswith(('Best ', 'Set ', 'Width limits', 'No feasible', 'Even axial'))]
                    info['result']['outcome'] = lines
                    best = re.match(r'Best \w+: ([-+0-9.eE]+)', lines[0]) if lines else None
                    info['result']['best_value'] = float(best.group(1)) if best else None
        return info


def run_result(out, cases):
    rows = []
    for c in cases:
        summary = read_json(os.path.join(out, c, 'summary.json'))
        if summary:
            rows.append({'case': c, 'path': rel(os.path.join(out, c)), 'status': summary.get('status'), 'error': summary.get('error'),
                         'stats': format_stats(summary) if summary.get('status') == 'ok' else []})
    image = os.path.join(out, 'summary.png')
    return {'type': 'run', 'cases': rows, 'image': rel(image) if os.path.isfile(image) else None}


def sweep_result(folder):
    points = []
    for f in sorted(glob.glob(os.path.join(folder, '*', 'summary.json'))):
        s = read_json(f)
        if s and 'sweep_value' in s:
            points.append({'case': os.path.basename(os.path.dirname(f)), 'path': rel(os.path.dirname(f)), 'value': s['sweep_value'],
                           'status': s.get('status'), 'error': s.get('error'), 'stats': format_stats(s) if s.get('status') == 'ok' else []})
    points.sort(key=lambda p: p['value'])
    images = [rel(os.path.join(folder, n)) for n in ('sweep.png', 'sweep_summary.png') if os.path.isfile(os.path.join(folder, n))]
    return {'type': 'sweep', 'points': points, 'images': images, 'folder': rel(folder)}


def optimize_progress(job):
    # stage of optimize.py from its output, and the 2D verification cases
    text = tail(job.log, 400)
    stage, fraction = 'Setting up', 0.02
    matches = STAGE.findall(text)
    if matches:
        name, r, R, i, n = matches[-1]
        i, n = int(i), max(int(n), 1)
        if name == 'Grid':
            stage, fraction = 'Fast model, grid: %d / %d designs' % (i, n), 0.05 + 0.6 * i / n
        else:
            stage, fraction = 'Fast model, refinement %s of %s' % (r, R), 0.65 + 0.2 * (int(r) - 1 + i / n) / int(R)
    cases = []
    if 'Verifying' in text:
        for folder in sorted(glob.glob(os.path.join(job.folder, 'verified', '*'))):
            if os.path.isdir(folder):
                cases.append(dict(name=os.path.basename(folder), **case_progress(folder, job.started)))
        stage = '2D verification'
        fraction = 0.85 + 0.15 * (sum(c['fraction'] for c in cases) / len(cases) if cases else 0.0)
    if 'Results in' in text:
        stage, fraction = 'Done', 1.0
    return {'cases': cases, 'fraction': min(fraction, 0.99 if job.proc.poll() is None else 1.0), 'stage': stage}


def optimize_result(folder, log=None):
    # outcome of optimize.py: charts, fast model and 2D values of the verified designs, the best design and its config lines
    rows = []
    path = os.path.join(folder, 'verified.csv')
    if os.path.isfile(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                num = lambda k: float(r[k]) if r.get(k) not in (None, '') else None
                rows.append({'design': r['design'], 'path': rel(os.path.join(folder, 'verified', r['design'])),
                             'fast_margin': num('fast_margin'), 'margin': num('2d_margin'), 'fast_T': num('fast_T_wall_max'), 'T': num('2d_T_wall_max'),
                             'fast_dp': num('fast_dp'), 'dp': num('2d_dp'), 'status': r.get('2d_status') or 'failed', 'note': r.get('2d_note') or ''})
    for row in rows:
        for k in ('fast_T', 'T'):
            if row[k] is not None:
                row[k] = Units.temperature(row[k])
        for k in ('fast_dp', 'dp'):
            if row[k] is not None:
                row[k] /= 1e5
    best_lines = []
    if log:
        lines = tail(log, 400).splitlines()
        if any(l.startswith('Best design') for l in lines):
            start = next(i for i, l in enumerate(lines) if l.startswith('Best design'))
            best_lines = [l for l in lines[start:] if not l.startswith(('Config lines', 'Results in'))]
    best_file = os.path.join(folder, 'best_design.txt')
    config_lines = open(best_file).read() if os.path.isfile(best_file) else ''
    images = [rel(os.path.join(folder, n)) for n in ('optimize.png', 'verified_summary.png') if os.path.isfile(os.path.join(folder, n))]
    data = [rel(os.path.join(folder, n)) for n in ('designs_fast.csv', 'verified.csv') if os.path.isfile(os.path.join(folder, n))]
    return {'type': 'optimize', 'folder': rel(folder), 'verified': rows, 'best': best_lines, 'config_lines': config_lines, 'images': images, 'data': data}


def browse(out):
    # result folders in an output folder: batch summary, cases and sweeps
    out = resolve(out)
    allowed_roots.add(os.path.realpath(out))
    entries = []
    if os.path.isfile(os.path.join(out, 'summary.png')):
        entries.append({'type': 'summary', 'name': 'Batch summary', 'path': rel(out), 'mtime': os.path.getmtime(os.path.join(out, 'summary.png'))})
    for folder in sorted(glob.glob(os.path.join(out, '*'))):
        if not os.path.isdir(folder):
            continue
        name = os.path.basename(folder)
        if os.path.isfile(os.path.join(folder, 'designs_fast.csv')):
            entries.append({'type': 'optimize', 'name': name, 'path': rel(folder), 'mtime': os.path.getmtime(os.path.join(folder, 'designs_fast.csv'))})
        elif os.path.isfile(os.path.join(folder, 'sweep.csv')):
            entries.append({'type': 'sweep', 'name': name, 'path': rel(folder), 'mtime': os.path.getmtime(os.path.join(folder, 'sweep.csv'))})
        elif os.path.isfile(os.path.join(folder, 'summary.json')):
            s = read_json(os.path.join(folder, 'summary.json')) or {}
            entries.append({'type': 'case', 'name': name, 'path': rel(folder), 'status': s.get('status'), 'mtime': os.path.getmtime(os.path.join(folder, 'summary.json')),
                            'T_wall_max': Units.temperature(s['T_wall_max']) if isinstance(s.get('T_wall_max'), (int, float)) else None,
                            'dp': s['dp_coolant'] / 1e5 if isinstance(s.get('dp_coolant'), (int, float)) else None})
    return {'out': rel(out), 'exists': os.path.isdir(out), 'entries': entries}


def case_detail(path):
    folder = resolve(path)
    if not allowed(folder):
        raise PermissionError(path)
    summary = read_json(os.path.join(folder, 'summary.json')) or {}
    images = [rel(os.path.join(folder, n)) for n in ('temperature.png', 'stress.png', 'heat_flux.png', 'pressure.png', 'halpha.png', 'reynolds.png', 'chamber_contour.png')
              if os.path.isfile(os.path.join(folder, n))]
    sections = sorted((f for f in glob.glob(os.path.join(folder, '*.png')) if os.path.basename(f)[:-4].isdigit()), key=lambda f: int(os.path.basename(f)[:-4]))
    return {'name': os.path.basename(folder), 'path': rel(folder), 'status': summary.get('status'), 'error': summary.get('error'),
            'stats': format_stats(summary) if summary.get('status') == 'ok' else [], 'images': images,
            'section_images': [rel(f) for f in sections], 'data': rel(os.path.join(folder, 'sim_data.csv')) if os.path.isfile(os.path.join(folder, 'sim_data.csv')) else None}


def start_run(body):
    configs = [resolve(c) for c in body.get('configs', [])]
    if not configs:
        raise ValueError('Select at least one config')
    for c in configs:
        if not (os.path.isfile(c) and c.endswith('.py')):
            raise ValueError('Config not found: ' + c)
    names = [case_name(c) for c in configs]
    if len(set(names)) != len(names):
        raise ValueError('Config file names must be unique, they name the output folders')
    out = resolve(body.get('out') or 'results')
    jobs_n = max(1, int(body.get('jobs') or 1))
    cmd = [sys.executable, os.path.join(REPO, 'run.py')] + configs + ['--out', out, '--jobs', str(jobs_n), '--logs']
    if body.get('section_images'):
        cmd.append('--section-images')
    if body.get('summary_only'):
        cmd.append('--summary-only')
    title = ('Single run: ' + names[0]) if len(names) == 1 else 'Batch: %d configs' % len(names)
    if body.get('summary_only'):
        title = 'Summary: ' + ', '.join(names)
    return Job('run', title, cmd, out, cases=names)


def start_optimize(body):
    config = resolve(body.get('config', ''))
    if not (os.path.isfile(config) and config.endswith('.py')):
        raise ValueError('Select a config')
    materials = [m for m in body.get('materials', []) if m in ('IN718', 'AlSi10Mg', 'CuCr1Zr')]
    modes = [m for m in body.get('modes', []) if m in ('constant_width', 'constant_fill')]
    if not materials or not modes:
        raise ValueError('Select at least one material and one channel width mode')

    def num(key, required=False):
        value = body.get(key)
        if value in (None, ''):
            if required:
                raise ValueError('Enter ' + key.replace('_', ' '))
            return None
        return float(value)

    out = resolve(body.get('out') or 'results')
    cmd = [sys.executable, os.path.join(REPO, 'optimize.py'), config, '--materials'] + materials + ['--modes'] + modes
    for flag, key, required in (('--max-dp', 'max_dp', True), ('--max-angle', 'max_angle', False), ('--min-rib', 'min_rib', True),
                                ('--min-channel-width', 'min_channel_width', True), ('--min-channel-height', 'min_channel_height', True),
                                ('--max-channel-height', 'max_channel_height', False), ('--wall', 'wall', False),
                                ('--max-coolant-wall-temp', 'max_coolant_wall_temp', False), ('--min-boil-margin', 'min_boil_margin', False)):
        value = num(key, required)
        if value is not None:
            cmd += [flag, repr(value)]
    if body.get('density') in ('coarse', 'normal', 'fine'):
        cmd += ['--density', body['density']]
    if body.get('longitudinal') in ('shell', 'wall'):
        cmd += ['--longitudinal', body['longitudinal']]
    if body.get('section_model') in ('grid', 'network'):
        cmd += ['--section-model', body['section_model']]
    cmd += ['--verify', str(int(num('verify') or 3)), '--jobs', str(max(1, int(num('jobs') or 1))), '--out', out]
    folder = os.path.join(out, 'optimize_' + case_name(config))
    return Job('optimize', 'Optimize: %s' % case_name(config), cmd, out, folder=folder)


def start_sweep(body):
    config = resolve(body.get('config', ''))
    if not (os.path.isfile(config) and config.endswith('.py')):
        raise ValueError('Select a config')
    param = body.get('param', 'pitch')
    if param not in ('pitch', 'angle'):
        raise ValueError('Unknown sweep parameter')
    num = lambda key, default=None: float(body[key]) if body.get(key) not in (None, '') else default
    v_max, max_dp = num('max'), num('max_dp')
    if v_max is None or max_dp is None:
        raise ValueError('Enter the maximum ' + param + ' and the maximum pressure drop')
    points, refine, jobs_n = int(num('points', 7)), int(num('refine', 3)), max(1, int(num('jobs', 1)))
    out = resolve(body.get('out') or 'results')
    cmd = [sys.executable, os.path.join(REPO, 'sweep.py'), config, '--param', param, '--max', repr(v_max), '--max-dp', repr(max_dp),
           '--min-channel-width', repr(num('min_channel', 0.0)), '--min-rib-width', repr(num('min_rib', 0.0)),
           '--points', str(points), '--refine', str(refine), '--jobs', str(jobs_n), '--out', out, '--logs']
    if num('max_coolant_wall_temp') is not None:
        cmd += ['--max-coolant-wall-temp', repr(num('max_coolant_wall_temp'))]
    if body.get('fresh'):
        cmd.append('--fresh')
    folder = os.path.join(out, 'sweep_%s_%s' % (case_name(config), param))
    return Job('sweep', 'Sweep: %s %s' % (case_name(config), param), cmd, out, folder=folder, params={'param': param, 'points': points, 'refine': refine})


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, code, body, content_type='application/json'):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path in ('/', '/index.html'):
                with open(os.path.join(STATIC, 'index.html'), 'rb') as f:
                    return self.send(200, f.read(), 'text/html; charset=utf-8')
            if url.path == '/api/state':
                return self.send(200, {'configs': list_configs(), 'cpus': os.cpu_count(), 'temperature_unit': TEMP, 'repo': REPO})
            if url.path == '/api/config':
                path = resolve(q.get('path', ''))
                if not (allowed(path) and path.endswith('.py') and os.path.isfile(path)):
                    return self.send(404, {'error': 'Config not found'})
                with open(path) as f:
                    return self.send(200, {'path': rel(path), 'source': f.read()})
            if url.path == '/api/jobs':
                with jobs_lock:
                    items = [j.summary() for j in sorted(jobs.values(), key=lambda j: -j.id)]
                return self.send(200, {'jobs': items})
            if url.path == '/api/job':
                job = jobs.get(int(q.get('id', 0)))
                return self.send(200, job.summary(detail=True)) if job else self.send(404, {'error': 'Unknown job'})
            if url.path == '/api/results':
                return self.send(200, browse(q.get('out') or 'results'))
            if url.path == '/api/case':
                return self.send(200, case_detail(q.get('path', '')))
            if url.path == '/api/optimize':
                folder = resolve(q.get('path', ''))
                if not allowed(folder):
                    return self.send(403, {'error': 'Not allowed'})
                return self.send(200, optimize_result(folder))
            if url.path == '/api/sweep':
                folder = resolve(q.get('path', ''))
                if not allowed(folder):
                    return self.send(403, {'error': 'Not allowed'})
                return self.send(200, sweep_result(folder))
            if url.path == '/file':
                path = resolve(q.get('path', ''))
                if not allowed(path) or not os.path.isfile(path):
                    return self.send(404, {'error': 'File not found'})
                types = {'.png': 'image/png', '.csv': 'text/csv; charset=utf-8', '.json': 'application/json', '.txt': 'text/plain; charset=utf-8'}
                ext = os.path.splitext(path)[1].lower()
                if ext not in types:
                    return self.send(403, {'error': 'File type not served'})
                with open(path, 'rb') as f:
                    return self.send(200, f.read(), types[ext])
            return self.send(404, {'error': 'Not found'})
        except PermissionError:
            return self.send(403, {'error': 'Not allowed'})
        except Exception as e:
            return self.send(500, {'error': repr(e)})

    def do_POST(self):
        url = urlparse(self.path)
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length) or b'{}')
            if url.path in ('/api/run', '/api/sweep', '/api/optimize'):
                job = {'/api/run': start_run, '/api/sweep': start_sweep, '/api/optimize': start_optimize}[url.path](body)
                with jobs_lock:
                    jobs[job.id] = job
                return self.send(200, {'id': job.id})
            if url.path == '/api/stop':
                job = jobs.get(int(body.get('id', 0)))
                if job:
                    job.stop()
                return self.send(200, {'ok': True})
            return self.send(404, {'error': 'Not found'})
        except ValueError as e:
            return self.send(400, {'error': str(e)})
        except Exception as e:
            return self.send(500, {'error': repr(e)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Browser interface for PyRocket runs and sweeps.')
    parser.add_argument('--port', type=int, default=8765, help='port of the local web server (default: 8765)')
    args = parser.parse_args()

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print('PyRocket interface running at http://localhost:%d  (Ctrl+C to stop)' % args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for job in jobs.values():
            job.stop()
        server.server_close()
