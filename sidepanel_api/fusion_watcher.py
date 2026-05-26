"""Background process: poll the active training run's test_logits.npz and
recompute the honest softmax fusion across 4 model slots.

Slots (no 'cur' abstraction � explicit by model name):
  - cnxxl:  our depth point cloud model (MotionCleanestLinXLQuatHead)
  - raw_c1: our cluster-cycle variant (MotionCleanestLinXLSTQNetC1)
  - dsn:    UMDR's depth model (Zhou et al. TPAMI'23), K modality
  - m:      UMDR's RGB model (Zhou et al. TPAMI'23), M modality

For cnxxl and raw_c1 we have BOTH a frozen snapshot AND a possibly-live
work_dir/test_logits.npz. The live file wins if the corresponding work_dir
log shows training is in progress (mtime is newer than the snapshot).

Cache is written to sidepanel_api/state/fusion_cache.json so server.py serves
it cheaply, and the published payload exposes:
  - solo: {cnxxl, raw_c1, dsn, m}
  - live: 'cnxxl' | 'raw_c1' | None        (which slot is being trained)
  - live_epoch: int                         (epoch of the live model)
  - fusion: {cnxxl_dsn, cnxxl_raw, cnxxl_dsn_raw, cnxxl_dsn_raw_m}
  - oracle: same keys
  - history: list of per-eval rows (deepest 60).
"""
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(HERE, 'state')
CACHE = os.path.join(STATE_DIR, 'fusion_cache.json')

# Work dirs whose test_logits.npz we read live.
LIVE_DIRS = {
    'cnxxl':  '/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head',
    # raw_c1 slot now points at the z-rotation augmentation fine-tune
    # of cnxxl. Slot keeps its name for sidepanel back-compat.
    'raw_c1': '/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head_rotaug_scratch',
}
SNAPSHOTS = {
    'cnxxl':  '/notebooks/Anemon/external_ckpts/cnxxl_ep141_91.29_test_logits.npz',
    'raw_c1': '/notebooks/Anemon/external_ckpts/cnxxl_ep141_91.29_test_logits.npz',
}
LOGITS_DSN = '/notebooks/Anemon/dsn_official_valid_logits.npz'
LOGITS_M = '/notebooks/Anemon/external_ckpts/umdr_M_test_logits.npz'

DSN_T = 9.5  # train-calibrated logit scale for DSN
HISTORY_LIMIT = 60


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def sig_of(p):
    m = re.search(r'class_(\d+)/subject(\d+)_r(\d+)', p)
    if not m:
        return p
    return f'class_{m.group(1)}/subject{m.group(2)}_r{m.group(3)}'


def load_logits(p):
    A = np.load(p, allow_pickle=True)
    if 'sigs' in A.files:
        sigs = np.array([str(s) if str(s).startswith('class_') else sig_of(str(s))
                         for s in A['sigs']])
    else:
        sigs = np.array([sig_of(str(s)) for s in A['paths']])
    return A['logits'], A['labels'], sigs


def load_epoch_marker(p):
    """Return the eval epoch baked into the npz (main.py writes this) or None."""
    try:
        A = np.load(p, allow_pickle=True)
        if 'epoch' in A.files:
            val = int(A['epoch'][0])
            return val if val > 0 else None
    except Exception:
        pass
    return None


def aligned(ref_sigs, ref_lab, logits, labels, sigs):
    """Align other logits to reference sample order. Ignores label mismatches;
    ref_lab is the source of truth."""
    by = {s: i for i, s in enumerate(sigs)}
    order = np.array([by[s] for s in ref_sigs])
    return logits[order]


def accuracy(probs, labels):
    return float((probs.argmax(1) == labels).mean() * 100.0)


def fuse_uniform(probs_list, labels):
    return accuracy(np.mean(probs_list, axis=0), labels)


def oracle_any(*probs_list, labels):
    cor = np.zeros_like(labels, dtype=bool)
    for p in probs_list:
        cor |= (p.argmax(1) == labels)
    return float(cor.mean() * 100.0)


def _resolve_slot_logits(name):
    """Return (path, is_live, epoch_marker) for a slot. Prefer live file
    if the live test_logits.npz is newer than the snapshot."""
    snap = SNAPSHOTS[name]
    live = os.path.join(LIVE_DIRS[name], 'test_logits.npz')
    snap_mt = os.path.getmtime(snap) if os.path.isfile(snap) else 0
    live_mt = os.path.getmtime(live) if os.path.isfile(live) else 0
    if live_mt > snap_mt:
        ep = load_epoch_marker(live)
        return live, True, ep
    return snap, False, None


def latest_live_slot():
    """Which of {cnxxl, raw_c1} has the freshest live test_logits.npz."""
    best = None
    for name in LIVE_DIRS:
        live = os.path.join(LIVE_DIRS[name], 'test_logits.npz')
        if not os.path.isfile(live):
            continue
        mt = os.path.getmtime(live)
        snap_mt = os.path.getmtime(SNAPSHOTS[name]) if os.path.isfile(SNAPSHOTS[name]) else 0
        if mt <= snap_mt:
            continue  # live not newer than snapshot
        if best is None or mt > best[1]:
            best = (name, mt, live)
    return best


def compute_fusion():
    """Resolve all 4 slots, compute solo + fusion + oracle. ref-aligned by cnxxl."""
    cnxxl_path, cnxxl_live, cnxxl_ep = _resolve_slot_logits('cnxxl')
    raw_path,   raw_live,   raw_ep   = _resolve_slot_logits('raw_c1')
    cnxxl_log, cnxxl_lab, cnxxl_sigs = load_logits(cnxxl_path)
    raw_log,   raw_lab,   raw_sigs   = load_logits(raw_path)

    # Use cnxxl as the reference order. raw_c1, dsn, m all align to it.
    ref_sigs, ref_lab = cnxxl_sigs, cnxxl_lab
    p_cnxxl = softmax(cnxxl_log)
    p_raw   = softmax(aligned(ref_sigs, ref_lab, raw_log, raw_lab, raw_sigs))

    dsn_log, dsn_lab, dsn_sigs = load_logits(LOGITS_DSN)
    p_dsn = softmax(aligned(ref_sigs, ref_lab, dsn_log, dsn_lab, dsn_sigs) * DSN_T)

    p_m = None
    if os.path.isfile(LOGITS_M):
        try:
            m_log, m_lab, m_sigs = load_logits(LOGITS_M)
            p_m = softmax(aligned(ref_sigs, ref_lab, m_log, m_lab, m_sigs))
        except Exception as e:
            print(f'[watcher] M align failed: {e}', flush=True)

    out = {
        'solo': {
            'cnxxl':  round(accuracy(p_cnxxl, ref_lab), 2),
            'raw_c1': round(accuracy(p_raw,   ref_lab), 2),
            'dsn':    round(accuracy(p_dsn,   ref_lab), 2),
        },
        'fusion': {
            'cnxxl_dsn':         round(fuse_uniform([p_cnxxl, p_dsn], ref_lab), 2),
            'cnxxl_raw':         round(fuse_uniform([p_cnxxl, p_raw], ref_lab), 2),
            'cnxxl_dsn_raw':     round(fuse_uniform([p_cnxxl, p_dsn, p_raw], ref_lab), 2),
        },
        'oracle': {
            'cnxxl_dsn':         round(oracle_any(p_cnxxl, p_dsn, labels=ref_lab), 2),
            'cnxxl_raw':         round(oracle_any(p_cnxxl, p_raw, labels=ref_lab), 2),
            'cnxxl_dsn_raw':     round(oracle_any(p_cnxxl, p_dsn, p_raw, labels=ref_lab), 2),
        },
    }
    if p_m is not None:
        out['solo']['m'] = round(accuracy(p_m, ref_lab), 2)
        out['fusion']['cnxxl_dsn_raw_m'] = round(fuse_uniform([p_cnxxl, p_dsn, p_raw, p_m], ref_lab), 2)
        out['oracle']['cnxxl_dsn_raw_m'] = round(oracle_any(p_cnxxl, p_dsn, p_raw, p_m, labels=ref_lab), 2)

    # Live-slot annotation.
    if cnxxl_live and raw_live:
        # both newer than snapshot � pick whichever is fresher.
        live_name = 'cnxxl' if os.path.getmtime(cnxxl_path) >= os.path.getmtime(raw_path) else 'raw_c1'
    elif cnxxl_live:
        live_name = 'cnxxl'
    elif raw_live:
        live_name = 'raw_c1'
    else:
        live_name = None
    out['live'] = live_name
    out['live_epoch'] = cnxxl_ep if live_name == 'cnxxl' else (raw_ep if live_name == 'raw_c1' else None)
    return out


def load_cache():
    if not os.path.isfile(CACHE):
        return None
    try:
        with open(CACHE) as f:
            return json.load(f)
    except Exception:
        return None


def write_cache(payload):
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = CACHE + '.tmpjson'
    with open(tmp, 'w') as f:
        json.dump(payload, f)
    os.replace(tmp, CACHE)


def append_history(latest):
    """Build a one-row summary keyed by live_epoch; replace same-epoch dupes."""
    prev = load_cache() or {}
    history = list(prev.get('history') or [])
    row = {
        'epoch': latest.get('live_epoch'),
        'ts': latest.get('ts'),
        'live': latest.get('live'),
        # solo of each slot at this tick
        'cnxxl':  latest['solo'].get('cnxxl'),
        'raw_c1': latest['solo'].get('raw_c1'),
        'dsn':    latest['solo'].get('dsn'),
        'm':      latest['solo'].get('m'),
        # fusion
        'cnxxl_dsn':         latest['fusion'].get('cnxxl_dsn'),
        'cnxxl_raw':         latest['fusion'].get('cnxxl_raw'),
        'cnxxl_dsn_raw':     latest['fusion'].get('cnxxl_dsn_raw'),
        'cnxxl_dsn_raw_m':   latest['fusion'].get('cnxxl_dsn_raw_m'),
        # oracle (3-way + 4-way most useful)
        'orc_cnxxl_dsn_raw': latest['oracle'].get('cnxxl_dsn_raw'),
        'orc_cnxxl_dsn_raw_m': latest['oracle'].get('cnxxl_dsn_raw_m'),
    }
    if history and history[-1].get('epoch') == row['epoch'] and history[-1].get('live') == row['live']:
        history[-1] = row
    else:
        history.append(row)
    history = history[-HISTORY_LIMIT:]
    merged = dict(latest)
    merged['history'] = history
    return merged


def step(last_mtime):
    live = latest_live_slot()
    if live is None:
        return last_mtime
    name, mtime, path = live
    if mtime == last_mtime:
        return last_mtime
    print(f'[watcher] {name} test_logits changed (mtime={mtime})', flush=True)
    try:
        fus = compute_fusion()
        fus['ts'] = time.strftime('%Y-%m-%d %H:%M:%S')
        merged = append_history(fus)
        write_cache(merged)
        print(f"[watcher] cache updated live={fus.get('live')} ep={fus.get('live_epoch')} "
              f"solo={fus['solo']} (history n={len(merged.get('history') or [])})",
              flush=True)
    except Exception as e:
        print(f'[watcher] fusion error: {e}', flush=True)
        return last_mtime
    return mtime


def main():
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    print(f'[watcher] polling live dirs every {interval}s: {list(LIVE_DIRS)}', flush=True)
    last = 0.0
    initial = latest_live_slot()
    if initial:
        # On startup, recompute only if cache is missing or stale.
        if not os.path.isfile(CACHE) or os.path.getmtime(CACHE) < initial[1]:
            last = step(0.0)
        else:
            last = initial[1]
    while True:
        try:
            last = step(last)
        except Exception as e:
            print(f'[watcher] step error: {e}', flush=True)
        time.sleep(interval)


if __name__ == '__main__':
    main()
