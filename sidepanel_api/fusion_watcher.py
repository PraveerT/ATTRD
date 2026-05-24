"""Background process: poll best_model.pt mtime; on change, refresh the
test-logits dump for the current stqnet_c1 run and recompute the honest
softmax fusion against cnxxlquat + DSN. Result cached as
sidepanel_api/state/fusion_cache.json so server.py can serve it cheaply.
"""
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(HERE, 'state')
CACHE = os.path.join(STATE_DIR, 'fusion_cache.json')
WATCH_DIR = '/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head_stqnet_c1'
BEST_PT = os.path.join(WATCH_DIR, 'best_model.pt')
LOGITS_CUR = os.path.join(WATCH_DIR, 'test_logits.npz')
# main.py writes test_logits.npz at the end of every test eval. The watcher
# fires on every change of that file (free CPU math, no GPU contention).
WATCH_FILE = LOGITS_CUR
LOGITS_CNXXL = '/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head/test_logits.npz'
LOGITS_DSN = '/notebooks/Anemon/dsn_official_valid_logits.npz'
LOGITS_UMDR_M = '/notebooks/Anemon/external_ckpts/umdr_M_test_logits.npz'
DUMP_SCRIPT = '/notebooks/Anemon/experiments/dump_raw_c1.py'

DSN_T = 9.5  # train-calibrated logit scale for DSN (multiply pre-softmax)


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
    """Return the eval epoch stored in the npz, or None if absent."""
    try:
        A = np.load(p, allow_pickle=True)
        if 'epoch' in A.files:
            val = int(A['epoch'][0])
            return val if val > 0 else None
    except Exception:
        pass
    return None


def aligned(ref_sigs, ref_lab, logits, labels, sigs):
    """Align logits to reference sample order. Labels are NOT checked because
    different dumps (e.g. path-derived vs splits-file derived) may disagree on
    a handful of NvGesture samples whose folder name differs from their actual
    label. ref_lab is always the source of truth."""
    by = {s: i for i, s in enumerate(sigs)}
    order = np.array([by[s] for s in ref_sigs])
    return logits[order]


def accuracy(probs, labels):
    return float((probs.argmax(1) == labels).mean() * 100.0)


def fuse_uniform(probs_list, labels):
    avg = np.mean(probs_list, axis=0)
    return accuracy(avg, labels)


def oracle_pair(p_a, p_b, labels):
    """Oracle: at each sample, take the model that was correct (either one)."""
    cor_a = p_a.argmax(1) == labels
    cor_b = p_b.argmax(1) == labels
    return float((cor_a | cor_b).mean() * 100.0)


def oracle_triple(p_a, p_b, p_c, labels):
    cor_a = p_a.argmax(1) == labels
    cor_b = p_b.argmax(1) == labels
    cor_c = p_c.argmax(1) == labels
    return float((cor_a | cor_b | cor_c).mean() * 100.0)


def dump_current():
    """main.py now writes test_logits.npz at the end of each test eval; we
    only need to check the file is present. Kept as a no-op so the rest of the
    watcher logic (which gates on dump success) still works."""
    return os.path.isfile(LOGITS_CUR)


def get_ckpt_epoch():
    """Read current best_model.pt's recorded epoch index. main.py stores it
    1-indexed (matches 'Epoch N, Test, Evaluation' log lines)."""
    try:
        import torch
        d = torch.load(BEST_PT, map_location='cpu')
        ep = d.get('epoch')
        return int(ep) if ep is not None else None
    except Exception:
        return None


def oracle_quad(p_a, p_b, p_c, p_d, labels):
    cor = (p_a.argmax(1) == labels) | (p_b.argmax(1) == labels) \
        | (p_c.argmax(1) == labels) | (p_d.argmax(1) == labels)
    return float(cor.mean() * 100.0)


def compute_fusion():
    """Load logits, align, compute solo + 2/3/4-way + oracle. Returns dict."""
    cur = load_logits(LOGITS_CUR)
    cnxxl = load_logits(LOGITS_CNXXL)
    dsn = load_logits(LOGITS_DSN)
    ref_log, ref_lab, ref_sigs = cur

    p_cur = softmax(ref_log)
    p_cnxxl = softmax(aligned(ref_sigs, ref_lab, *cnxxl))
    p_dsn = softmax(aligned(ref_sigs, ref_lab, *dsn) * DSN_T)

    # UMDR-M (rgb modality) — optional, only if file present.
    p_m = None
    if os.path.isfile(LOGITS_UMDR_M):
        try:
            m = load_logits(LOGITS_UMDR_M)
            p_m = softmax(aligned(ref_sigs, ref_lab, *m))
        except Exception as e:
            print(f'[watcher] UMDR-M align failed: {e}', flush=True)
            p_m = None

    out = {
        'solo': {
            'cur': round(accuracy(p_cur, ref_lab), 2),
            'cnxxl': round(accuracy(p_cnxxl, ref_lab), 2),
            'dsn': round(accuracy(p_dsn, ref_lab), 2),
        },
        'fusion': {
            'cnxxl_cur': round(fuse_uniform([p_cnxxl, p_cur], ref_lab), 2),
            'cnxxl_dsn': round(fuse_uniform([p_cnxxl, p_dsn], ref_lab), 2),
            'cnxxl_dsn_cur': round(fuse_uniform([p_cnxxl, p_dsn, p_cur], ref_lab), 2),
        },
        'oracle': {
            'cnxxl_cur': round(oracle_pair(p_cnxxl, p_cur, ref_lab), 2),
            'cnxxl_dsn': round(oracle_pair(p_cnxxl, p_dsn, ref_lab), 2),
            'cnxxl_dsn_cur': round(oracle_triple(p_cnxxl, p_dsn, p_cur, ref_lab), 2),
        },
    }
    if p_m is not None:
        out['solo']['m'] = round(accuracy(p_m, ref_lab), 2)
        out['fusion']['cnxxl_m'] = round(fuse_uniform([p_cnxxl, p_m], ref_lab), 2)
        out['fusion']['cnxxl_dsn_m'] = round(fuse_uniform([p_cnxxl, p_dsn, p_m], ref_lab), 2)
        out['fusion']['cnxxl_dsn_m_cur'] = round(fuse_uniform([p_cnxxl, p_dsn, p_m, p_cur], ref_lab), 2)
        out['oracle']['cnxxl_dsn_m'] = round(oracle_triple(p_cnxxl, p_dsn, p_m, ref_lab), 2)
        out['oracle']['cnxxl_dsn_m_cur'] = round(oracle_quad(p_cnxxl, p_dsn, p_m, p_cur, ref_lab), 2)
    return out


HISTORY_LIMIT = 60


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
    """Append a single-line per-epoch summary row to the history list and
    persist back to the cache. Returns the merged cache dict."""
    prev = load_cache() or {}
    history = list(prev.get('history') or [])
    row = {
        'epoch': latest.get('epoch'),
        'ts': latest.get('ts'),
        'cur': latest['solo']['cur'],
        'cnxxl_cur': latest['fusion']['cnxxl_cur'],
        'cnxxl_dsn': latest['fusion']['cnxxl_dsn'],
        'three_way': latest['fusion']['cnxxl_dsn_cur'],
        'orc_two': latest['oracle']['cnxxl_cur'],
        'orc_three': latest['oracle']['cnxxl_dsn_cur'],
    }
    if 'cnxxl_dsn_m_cur' in latest.get('fusion', {}):
        row['four_way'] = latest['fusion']['cnxxl_dsn_m_cur']
        row['cnxxl_dsn_m'] = latest['fusion']['cnxxl_dsn_m']
        row['orc_four'] = latest['oracle'].get('cnxxl_dsn_m_cur')
    # Skip dup: same epoch as last row → replace, not append.
    if history and history[-1].get('epoch') == row['epoch']:
        history[-1] = row
    else:
        history.append(row)
    history = history[-HISTORY_LIMIT:]
    merged = dict(latest)
    merged['history'] = history
    return merged


def step(last_mtime):
    if not os.path.isfile(WATCH_FILE):
        return last_mtime
    mtime = os.path.getmtime(WATCH_FILE)
    if mtime == last_mtime:
        return last_mtime
    print(f'[watcher] test_logits.npz changed (mtime={mtime})', flush=True)
    try:
        # Prefer the epoch marker baked into test_logits.npz (= eval epoch).
        # Falls back to best_model.pt's recorded epoch (only useful while
        # tests pre-date the marker patch).
        ep = load_epoch_marker(WATCH_FILE) or get_ckpt_epoch()
        fus = compute_fusion()
        fus['epoch'] = ep
        fus['ts'] = time.strftime('%Y-%m-%d %H:%M:%S')
        merged = append_history(fus)
        write_cache(merged)
        print(f'[watcher] cache updated for ep={ep}: {fus["solo"]} '
              f'(history n={len(merged.get("history") or [])})', flush=True)
    except Exception as e:
        print(f'[watcher] fusion error: {e}', flush=True)
        return last_mtime
    return mtime


def main():
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    print(f'[watcher] watching {WATCH_FILE} every {interval}s', flush=True)
    last = 0.0
    if os.path.isfile(WATCH_FILE):
        # On startup, prime the cache only if it's missing/stale.
        if not os.path.isfile(CACHE) or os.path.getmtime(CACHE) < os.path.getmtime(WATCH_FILE):
            last = step(0.0)
        else:
            last = os.path.getmtime(WATCH_FILE)
    while True:
        try:
            last = step(last)
        except Exception as e:
            print(f'[watcher] step error: {e}', flush=True)
        time.sleep(interval)


if __name__ == '__main__':
    main()
