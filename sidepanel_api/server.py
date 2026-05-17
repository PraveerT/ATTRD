"""Tiny stdlib HTTP server: serves PWA static files + /api/status JSON.

JSON shape:
  {
    "ts": "2026-05-17 07:42:11",
    "run": "bdn_buf1_train",
    "log": "/notebooks/PMamba/.../bdn_buf1_train.log",
    "gpu": {"used_gb": 11.2, "total_gb": 47.6, "util_pct": 87},
    "ram": {"used_gb": 14, "total_gb": 64},
    "disk": {"used_gb": 22.1, "total_gb": 50},
    "epochs": [{"ep": 10, "tr_acc": 12.3, "tr_loss": 3.27, "te_p1": 69.5, "te_p5": 94.4}, ...],
    "best": {"ep": 50, "p1": 85.68},
    "now": {"ep": 70, "batch": "12/131", "elapsed_min": 4.3}
  }
"""
import json, os, re, subprocess, time, glob
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, 'static')
LOG_GLOB = '/notebooks/PMamba/experiments/work_dir/*.log'

_cache = {'ts': 0, 'data': None}
_CACHE_SEC = 5


def latest_log():
    files = sorted(glob.glob(LOG_GLOB), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def parse_log(path):
    """Re-implements tg_messages.sh awk, returns dict."""
    epochs = []
    cur_ep = None
    cur_batch = None
    ta = None
    tl = None
    best = None
    try:
        with open(path, 'r', errors='replace') as f:
            for line in f:
                m = re.search(r'Mean training acc:\s+(\d+(?:\.\d+)?)', line)
                if m: ta = float(m.group(1))
                m = re.search(r'Mean training loss:\s+(\d+(?:\.\d+)?)', line)
                if m: tl = float(m.group(1))
                m = re.search(r'Training epoch:\s+(\d+)', line)
                if m: cur_ep = int(m.group(1))
                m = re.search(r'(\d+)/(\d+) \[', line)
                if m: cur_batch = f"{m.group(1)}/{m.group(2)}"
                m = re.search(r'Epoch (\d+), Test, Evaluation: prec1 (\d+(?:\.\d+)?), prec5 (\d+(?:\.\d+)?)', line)
                if m:
                    ep = int(m.group(1))
                    p1 = float(m.group(2))
                    p5 = float(m.group(3))
                    epochs.append({
                        'ep': ep, 'tr_acc': ta, 'tr_loss': tl,
                        'te_p1': round(p1, 2), 'te_p5': round(p5, 2),
                    })
                    if best is None or p1 > best['p1']:
                        best = {'ep': ep, 'p1': round(p1, 2)}
    except FileNotFoundError:
        pass
    return {'epochs': epochs[-40:], 'best': best, 'now': {'ep': cur_ep, 'batch': cur_batch}}


def gpu_stats():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'], text=True, timeout=4).strip().splitlines()[0]
        used, total, util = [float(x.strip()) for x in out.split(',')]
        return {'used_gb': round(used / 1024, 1), 'total_gb': round(total / 1024, 1), 'util_pct': int(util)}
    except Exception:
        return None


def ram_stats():
    try:
        out = subprocess.check_output(['free', '-g'], text=True, timeout=4)
        for line in out.splitlines():
            if line.startswith('Mem:'):
                p = line.split()
                return {'used_gb': int(p[2]), 'total_gb': int(p[1])}
    except Exception:
        return None


def disk_stats():
    try:
        nb = int(os.getxattr('/notebooks', 'ceph.dir.rbytes'))
        return {'used_gb': round(nb / 1e9, 1), 'total_gb': 50}
    except Exception:
        return None


def build_status():
    log = latest_log()
    parsed = parse_log(log) if log else {'epochs': [], 'best': None, 'now': {}}
    return {
        'ts': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'run': os.path.basename(log).replace('.log', '') if log else None,
        'log': log,
        'gpu': gpu_stats(),
        'ram': ram_stats(),
        'disk': disk_stats(),
        **parsed,
    }


def cached_status():
    now = time.time()
    if _cache['data'] is None or (now - _cache['ts']) > _CACHE_SEC:
        _cache['data'] = build_status()
        _cache['ts'] = now
    return _cache['data']


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):  # quieter
        pass

    def _send(self, code, body, ctype='application/json; charset=utf-8'):
        body_b = body if isinstance(body, (bytes, bytearray)) else body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body_b)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body_b)

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path == '/api/status':
            self._send(200, json.dumps(cached_status()))
            return
        if path == '/' or path == '':
            path = '/index.html'
        safe = os.path.normpath(path).lstrip(os.sep)
        full = os.path.join(STATIC, safe)
        if not full.startswith(STATIC) or not os.path.isfile(full):
            self._send(404, '{"error":"not found"}')
            return
        ctype = {
            '.html': 'text/html; charset=utf-8',
            '.js':   'application/javascript; charset=utf-8',
            '.json': 'application/json; charset=utf-8',
            '.css':  'text/css; charset=utf-8',
            '.svg':  'image/svg+xml',
            '.png':  'image/png',
        }.get(os.path.splitext(full)[1].lower(), 'application/octet-stream')
        with open(full, 'rb') as f:
            self._send(200, f.read(), ctype)


def main():
    port = int(os.environ.get('SIDEPANEL_PORT', 8765))
    srv = ThreadingHTTPServer(('0.0.0.0', port), Handler)
    print(f'[sidepanel] serving on 0.0.0.0:{port}', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
