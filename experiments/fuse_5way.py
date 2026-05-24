"""5-way oracle + fusion: cnxxlquat, DSN, canonical, canonical+C1, raw+C1."""
import re, numpy as np

PATHS = {
    'cnxxlquat': '/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head/test_logits.npz',
    'canonical': '/notebooks/Anemon/experiments/work_dir/cn_xxl_canonical/test_logits.npz',
    'canon_c1':  '/notebooks/Anemon/experiments/work_dir/cn_xxl_canonical_stqnet_c1/test_logits.npz',
    'raw_c1':    '/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head_stqnet_c1/test_logits.npz',
    'dsn':       '/notebooks/Anemon/dsn_official_valid_logits.npz',
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

# Align to cnxxlquat order.
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
        assert (lab[order] == ref_lab).all(), f'label mismatch for {name}'

lab = ref_lab
print('solo accs:')
for name in PATHS:
    log = data[name][0]
    pred = log.argmax(1)
    acc = (pred == lab).mean() * 100
    n_wrong = (pred != lab).sum()
    print(f'  {name:<12} {acc:5.2f}%  wrong={n_wrong}/482')

# Oracle: at least one is correct.
preds = {name: data[name][0].argmax(1) for name in PATHS}
wrongs = {name: (preds[name] != lab) for name in PATHS}

def oracle(names):
    shared = wrongs[names[0]].copy()
    for n in names[1:]: shared &= wrongs[n]
    return (~shared).mean() * 100

print('\noracle (perfect router):')
print(f'  cnxxl+raw_c1            : {oracle(["cnxxlquat","raw_c1"]):.2f}%')
print(f'  cnxxl+canon_c1          : {oracle(["cnxxlquat","canon_c1"]):.2f}%')
print(f'  cnxxl+DSN               : {oracle(["cnxxlquat","dsn"]):.2f}%')
print(f'  cnxxl+DSN+raw_c1        : {oracle(["cnxxlquat","dsn","raw_c1"]):.2f}%')
print(f'  cnxxl+DSN+canon_c1      : {oracle(["cnxxlquat","dsn","canon_c1"]):.2f}%')
print(f'  cnxxl+DSN+canon+canon_c1: {oracle(["cnxxlquat","dsn","canonical","canon_c1"]):.2f}%')
print(f'  cnxxl+DSN+canon+canon_c1+raw_c1: {oracle(list(PATHS)):.2f}%')

# Pairwise saves analysis vs cnxxlquat.
print('\nvs cnxxlquat (43 wrong):')
for name in PATHS:
    if name == 'cnxxlquat': continue
    saves = (wrongs['cnxxlquat'] & ~wrongs[name]).sum()
    shared = (wrongs['cnxxlquat'] & wrongs[name]).sum()
    print(f'  {name:<12} saves={saves}  shared_wrong={shared}')

# Honest fusion attempts.
sm = {name: softmax(data[name][0]) for name in PATHS}
sm_dsn_T = softmax(data['dsn'][0] * 9.5)
sm['dsn_T'] = sm_dsn_T

def f(name, p):
    print(f'  {name:<40} {(p.argmax(1)==lab).mean()*100:.2f}%')

print('\nhonest fusion (uniform softmax):')
f('cnxxl+raw_c1                /2', (sm['cnxxlquat']+sm['raw_c1'])/2)
f('cnxxl+canon_c1              /2', (sm['cnxxlquat']+sm['canon_c1'])/2)
f('cnxxl+DSN(T=9.5)            /2', (sm['cnxxlquat']+sm_dsn_T)/2)
f('cnxxl+DSN+raw_c1            /3', (sm['cnxxlquat']+sm_dsn_T+sm['raw_c1'])/3)
f('cnxxl+DSN+canon_c1          /3', (sm['cnxxlquat']+sm_dsn_T+sm['canon_c1'])/3)
f('cnxxl+raw_c1+canon_c1       /3', (sm['cnxxlquat']+sm['raw_c1']+sm['canon_c1'])/3)
f('cnxxl+DSN+raw_c1+canon_c1   /4', (sm['cnxxlquat']+sm_dsn_T+sm['raw_c1']+sm['canon_c1'])/4)
f('all 5 (cnxxl+DSN+canon+canon_c1+raw_c1)/5',
  (sm['cnxxlquat']+sm_dsn_T+sm['canonical']+sm['canon_c1']+sm['raw_c1'])/5)
