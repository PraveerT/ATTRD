"""Inspect ST-QNet mechanism usage: rigidity_proj norm, cycle_proj_out norm."""
import torch, os

base = '/notebooks/Anemon/experiments/work_dir/cn_xxl_canonical_stqnet'
for fname in sorted(os.listdir(base)):
    if not fname.startswith('epoch') or not fname.endswith('_model.pt'):
        continue
    ep = int(fname.replace('epoch', '').replace('_model.pt', ''))
    if ep % 10 != 0 and ep != 5 and ep != 75:
        continue
    c = torch.load(os.path.join(base, fname), map_location='cpu')
    sd = c['model_state_dict']
    rigid_w = sd.get('rigidity_proj.weight')
    rigid_b = sd.get('rigidity_proj.bias')
    cycle_w = sd.get('cycle_proj_out.weight')
    cycle_b = sd.get('cycle_proj_out.bias')
    cycle_in_w = sd.get('cycle_proj_in.weight')
    print(f'ep{ep:3d}  rigid_proj norm={rigid_w.norm():.4f} (max={rigid_w.abs().max():.4f}) '
          f'rigid_b norm={rigid_b.norm():.4f}  '
          f'cycle_out norm={cycle_w.norm():.4f} (max={cycle_w.abs().max():.4f})  '
          f'cycle_in norm={cycle_in_w.norm():.4f}')

# Also: check best_model
b = torch.load(os.path.join(base, 'best_model.pt'), map_location='cpu')
sd = b['model_state_dict']
print(f'\nbest_model ep={b.get("epoch")} '
      f'rigid_proj norm={sd["rigidity_proj.weight"].norm():.4f} '
      f'cycle_out norm={sd["cycle_proj_out.weight"].norm():.4f}')
