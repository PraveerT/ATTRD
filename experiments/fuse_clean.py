"""Honest fusion analysis: cnxxlquat + DSN + raw+C1 only.

All ckpts test-best by default. Pass --train-best to use train-best ckpts.
"""
import argparse, re, numpy as np

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-best', action='store_true',
                    help='use *_train_best.npz dumps instead of test-best best_model.pt dumps')
    args = ap.parse_args()

    suffix = '_train_best' if args.train_best else ''
    PATHS = {
        'cnxxl':  f'/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head/test_logits{suffix}.npz',
        'raw_c1': f'/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head_stqnet_c1/test_logits{suffix}.npz',
        'dsn':    '/notebooks/Anemon/dsn_official_valid_logits.npz',
    }

    ref = None
    data = {}
    for name, p in PATHS.items():
        log, lab, sigs = load(p)
        if ref is None:
            ref, ref_lab = sigs, lab
            data[name] = (log, lab)
        else:
            by = {s: i for i, s in enumerate(sigs)}
            order = np.array([by[s] for s in ref])
            data[name] = (log[order], lab[order])
            assert (data[name][1] == ref_lab).all()

    lab = ref_lab
    print(f'ckpt selection: {"TRAIN-BEST" if args.train_best else "TEST-BEST"}')
    print('solo:')
    for name in PATHS:
        log = data[name][0]
        acc = (log.argmax(1) == lab).mean() * 100
        wrong = (log.argmax(1) != lab).sum()
        print(f'  {name:<8} {acc:5.2f}%  wrong={wrong}/482')

    # Pairwise saves vs cnxxlquat
    print('\nvs cnxxlquat:')
    cn_pred = data['cnxxl'][0].argmax(1)
    cn_wrong = cn_pred != lab
    for name in PATHS:
        if name == 'cnxxl': continue
        pred = data[name][0].argmax(1)
        wrong = pred != lab
        saves = (cn_wrong & ~wrong).sum()
        shared = (cn_wrong & wrong).sum()
        print(f'  {name:<8} saves={saves}  shared_wrong={shared}')

    # Honest fusion
    sm = {name: softmax(data[name][0]) for name in PATHS}
    sm_dsn_T = softmax(data['dsn'][0] * 9.5)

    def f(name, p):
        print(f'  {name:<40} {(p.argmax(1)==lab).mean()*100:.2f}%')

    print('\nhonest uniform softmax fusion:')
    f('cnxxl solo',                              sm['cnxxl'])
    f('cnxxl + raw_c1 /2',                       (sm['cnxxl']+sm['raw_c1'])/2)
    f('cnxxl + DSN(T=9.5) /2',                   (sm['cnxxl']+sm_dsn_T)/2)
    f('cnxxl + DSN + raw_c1 /3',                 (sm['cnxxl']+sm_dsn_T+sm['raw_c1'])/3)

    # Oracle (perfect router) for reference
    preds = {n: data[n][0].argmax(1) for n in PATHS}
    wrongs = {n: preds[n] != lab for n in PATHS}
    def oracle(ns):
        s = wrongs[ns[0]].copy()
        for n in ns[1:]: s &= wrongs[n]
        return (~s).mean()*100
    print('\noracle (perfect router):')
    print(f'  cnxxl + raw_c1        : {oracle(["cnxxl","raw_c1"]):.2f}%')
    print(f'  cnxxl + DSN           : {oracle(["cnxxl","dsn"]):.2f}%')
    print(f'  cnxxl + DSN + raw_c1  : {oracle(["cnxxl","dsn","raw_c1"]):.2f}%')


if __name__ == '__main__':
    main()
