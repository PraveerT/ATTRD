"""Honest train-best fusion -- ckpt selection by train acc only."""
import re, numpy as np

PATHS = {
    'cnxxl_tb':       '/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head/test_logits_train_best.npz',
    'canonical_tb':   '/notebooks/Anemon/experiments/work_dir/cn_xxl_canonical/test_logits_train_best.npz',
    'canon_c1_tb':    '/notebooks/Anemon/experiments/work_dir/cn_xxl_canonical_stqnet_c1_v2/test_logits_train_best.npz',
    'raw_c1_tb':      '/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head_stqnet_c1/test_logits_train_best.npz',
    'dsn':            '/notebooks/Anemon/dsn_official_valid_logits.npz',
}

def softmax(x, ax=-1):
    x = x - x.max(ax, keepdims=True); e = np.exp(x)
    return e / e.sum(ax, keepdims=True)

def sig_of(p):
    m = re.search(r'class_(\d+)/subject(\d+)_r(\d+)', p)
    return f'class_{m.group(1)}/subject{m.group(2)}_r{m.group(3)}'

def load(p):
    A = np.load(p, allow_pickle=True)
    if 'sigs' in A.files:
        sigs = np.array([str(s) if str(s).startswith('class_') else sig_of(str(s)) for s in A['sigs']])
    else:
        sigs = np.array([sig_of(str(s)) for s in A['paths']])
    return A['logits'], A['labels'], sigs

ref_sigs = None
data = {}
for name, p in PATHS.items():
    log, lab, sigs = load(p)
    if ref_sigs is None:
        ref_sigs, ref_lab = sigs, lab
        data[name] = (log, lab, sigs)
    else:
        by_sig = {s: i for i, s in enumerate(sigs)}
        order = np.array([by_sig[s] for s in ref_sigs])
        data[name] = (log[order], lab[order], ref_sigs)
        assert (lab[order] == ref_lab).all(), f'label mismatch {name}'

lab = ref_lab
print('train-best ckpt solo accs:')
for name in PATHS:
    pred = data[name][0].argmax(1)
    print(f'  {name:<14} {(pred==lab).mean()*100:5.2f}%  wrong={(pred!=lab).sum()}/482')

sm = {name: softmax(data[name][0]) for name in PATHS}
sm['dsn_T9.5'] = softmax(data['dsn'][0] * 9.5)

def f(name, p):
    print(f'  {name:<46} {(p.argmax(1)==lab).mean()*100:.2f}%')

print('\nhonest fusion (uniform softmax, train-best ckpts only):')
f('cnxxl_tb + DSN(T=9.5) /2',                 (sm['cnxxl_tb']+sm['dsn_T9.5'])/2)
f('cnxxl_tb + raw_c1_tb /2',                  (sm['cnxxl_tb']+sm['raw_c1_tb'])/2)
f('cnxxl_tb + DSN + raw_c1_tb /3',            (sm['cnxxl_tb']+sm['dsn_T9.5']+sm['raw_c1_tb'])/3)
f('cnxxl_tb + DSN + canon_c1_tb /3',          (sm['cnxxl_tb']+sm['dsn_T9.5']+sm['canon_c1_tb'])/3)
f('cnxxl_tb + DSN + canonical_tb /3',         (sm['cnxxl_tb']+sm['dsn_T9.5']+sm['canonical_tb'])/3)
f('cnxxl_tb + DSN + raw_c1_tb + canon_c1_tb /4',
  (sm['cnxxl_tb']+sm['dsn_T9.5']+sm['raw_c1_tb']+sm['canon_c1_tb'])/4)
f('all 5 train-best /5',
  (sm['cnxxl_tb']+sm['dsn_T9.5']+sm['canonical_tb']+sm['canon_c1_tb']+sm['raw_c1_tb'])/5)

# Oracle for reference (still test-info-free since saves count only)
preds = {n: data[n][0].argmax(1) for n in PATHS}
wrongs = {n: preds[n] != lab for n in PATHS}
def oracle(ns):
    s = wrongs[ns[0]].copy()
    for n in ns[1:]: s &= wrongs[n]
    return (~s).mean()*100
print('\noracle (perfect router, train-best ckpts):')
print(f'  cnxxl+raw_c1            : {oracle(["cnxxl_tb","raw_c1_tb"]):.2f}%')
print(f'  cnxxl+DSN               : {oracle(["cnxxl_tb","dsn"]):.2f}%')
print(f'  cnxxl+DSN+raw_c1        : {oracle(["cnxxl_tb","dsn","raw_c1_tb"]):.2f}%')
print(f'  cnxxl+DSN+canon+canon_c1+raw_c1 : {oracle(list(PATHS)):.2f}%')
