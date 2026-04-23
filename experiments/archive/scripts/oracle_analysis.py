"""Oracle + late fusion analysis from two branches.

Oracle = upper bound (either branch correct).
Late fusion = weighted average of softmax outputs.
"""
import sys
sys.path.insert(0, '/notebooks/PMamba/experiments')

import torch
import numpy as np
from collections import defaultdict

# --- Load PMamba branch ---
from models.motion import Motion
from nvidia_dataloader import NvidiaLoader

pmamba_model = Motion(num_classes=25, pts_size=256, knn=[32, 24, 48, 24], topk=8).cuda()
ckpt_p = torch.load('work_dir/pmamba_branch/epoch110_model.pt', map_location='cpu')
state = ckpt_p.get('model_state_dict', ckpt_p.get('model', ckpt_p))
pmamba_model.load_state_dict(state, strict=False)
pmamba_model.eval()
print("PMamba loaded")

# --- Load Quaternion branch ---
from models.reqnn_motion import BearingQCCFeatureMotion

quat_model = BearingQCCFeatureMotion(
    num_classes=25, pts_size=256, hidden_dims=[64, 256],
    dropout=0.05, edgeconv_k=20, merge_eps=1e-6,
    so3_weight=0.0, rotation_sigma=0.3, bearing_knn_k=10,
    qcc_weight=0.1,
).cuda()
ckpt_q = torch.load('work_dir/quaternion_branch/epoch112_model.pt', map_location='cpu')
state_q = ckpt_q.get('model_state_dict', ckpt_q.get('model', ckpt_q))
quat_model.load_state_dict(state_q, strict=False)
quat_model.eval()
print("Quaternion loaded")

# --- Test data ---
# PMamba uses NvidiaLoader
pmamba_loader = NvidiaLoader(framerate=32, phase='test')
# Quaternion uses NvidiaQuaternionQCCParityLoader
from nvidia_dataloader import NvidiaQuaternionQCCParityLoader
quat_loader = NvidiaQuaternionQCCParityLoader(framerate=32, phase='test', return_correspondence=False)

n_test = len(pmamba_loader)
print(f"Test samples: {n_test}")
assert len(quat_loader) == n_test

pmamba_correct = np.zeros(n_test, dtype=bool)
quat_correct = np.zeros(n_test, dtype=bool)
pmamba_preds = np.zeros(n_test, dtype=int)
quat_preds = np.zeros(n_test, dtype=int)
labels = np.zeros(n_test, dtype=int)
all_pmamba_logits = []
all_quat_logits = []

with torch.no_grad():
    for i in range(n_test):
        # PMamba
        sample_p, label_p, _ = pmamba_loader[i]
        if isinstance(sample_p, dict):
            inp_p = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v for k, v in sample_p.items()}
        else:
            inp_p = sample_p.unsqueeze(0).cuda()
        # 3x TTA like eval loop
        outs_p = [pmamba_model(inp_p) for _ in range(3)]
        out_p = torch.stack(outs_p).mean(dim=0)
        pred_p = out_p.argmax(dim=1).item()

        # Quaternion
        sample_q, label_q, _ = quat_loader[i]
        if isinstance(sample_q, dict):
            inp_q = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v for k, v in sample_q.items()}
        else:
            inp_q = sample_q.unsqueeze(0).cuda()
        outs_q = [quat_model(inp_q) for _ in range(3)]
        out_q = torch.stack(outs_q).mean(dim=0)
        pred_q = out_q.argmax(dim=1).item()

        label = int(label_p)
        labels[i] = label
        pmamba_preds[i] = pred_p
        quat_preds[i] = pred_q
        pmamba_correct[i] = (pred_p == label)
        quat_correct[i] = (pred_q == label)
        all_pmamba_logits.append(out_p.cpu())
        all_quat_logits.append(out_q.cpu())

        if i % 50 == 0:
            print(f"  {i}/{n_test}...")

all_pmamba_logits = torch.cat(all_pmamba_logits, dim=0)  # (n_test, 25)
all_quat_logits = torch.cat(all_quat_logits, dim=0)

# --- Analysis ---
pmamba_acc = pmamba_correct.mean() * 100
quat_acc = quat_correct.mean() * 100
oracle_correct = pmamba_correct | quat_correct
oracle_acc = oracle_correct.mean() * 100
both_correct = pmamba_correct & quat_correct
both_wrong = ~pmamba_correct & ~quat_correct
only_pmamba = pmamba_correct & ~quat_correct
only_quat = ~pmamba_correct & quat_correct

print(f"\n{'='*50}")
print(f"PMamba accuracy:     {pmamba_acc:.2f}% ({pmamba_correct.sum()}/{n_test})")
print(f"Quaternion accuracy: {quat_acc:.2f}% ({quat_correct.sum()}/{n_test})")
print(f"Oracle accuracy:     {oracle_acc:.2f}% ({oracle_correct.sum()}/{n_test})")
print(f"{'='*50}")
print(f"Both correct:        {both_correct.sum()} ({both_correct.mean()*100:.1f}%)")
print(f"Only PMamba correct: {only_pmamba.sum()} ({only_pmamba.mean()*100:.1f}%)")
print(f"Only Quaternion:     {only_quat.sum()} ({only_quat.mean()*100:.1f}%)")
print(f"Both wrong:          {both_wrong.sum()} ({both_wrong.mean()*100:.1f}%)")
print(f"{'='*50}")
print(f"Complementarity:     {only_pmamba.sum() + only_quat.sum()} samples where exactly one is right")
print(f"Headroom over best:  {oracle_acc - max(pmamba_acc, quat_acc):.2f}%")

# --- Late Fusion: sweep alpha for weighted softmax average ---
print(f"\n{'='*50}")
print("Late Fusion: alpha * PMamba_softmax + (1-alpha) * Quat_softmax")
print(f"{'='*50}")

pmamba_probs = torch.softmax(all_pmamba_logits, dim=1)
quat_probs = torch.softmax(all_quat_logits, dim=1)
labels_t = torch.tensor(labels, dtype=torch.long)

best_alpha = 0
best_fusion_acc = 0
print(f"{'Alpha':>7} {'Accuracy':>10} {'Correct':>9}")
for alpha_int in range(0, 105, 5):
    alpha = alpha_int / 100.0
    fused = alpha * pmamba_probs + (1 - alpha) * quat_probs
    fused_preds = fused.argmax(dim=1)
    correct = (fused_preds == labels_t).sum().item()
    acc = correct / n_test * 100
    print(f"{alpha:>7.2f} {acc:>9.2f}% {correct:>6}/{n_test}")
    if acc > best_fusion_acc:
        best_fusion_acc = acc
        best_alpha = alpha

print(f"\nBest late fusion: alpha={best_alpha:.2f} → {best_fusion_acc:.2f}%")
print(f"PMamba alone: {pmamba_acc:.2f}%")
print(f"Improvement:  +{best_fusion_acc - pmamba_acc:.2f}%")
print(f"Oracle ceiling: {oracle_acc:.2f}%")

# Also try logit-level fusion (before softmax)
print(f"\n{'='*50}")
print("Logit-level fusion: alpha * PMamba_logits + (1-alpha) * Quat_logits")
print(f"{'='*50}")
best_alpha_logit = 0
best_logit_acc = 0
for alpha_int in range(0, 105, 5):
    alpha = alpha_int / 100.0
    fused = alpha * all_pmamba_logits + (1 - alpha) * all_quat_logits
    fused_preds = fused.argmax(dim=1)
    correct = (fused_preds == labels_t).sum().item()
    acc = correct / n_test * 100
    if alpha_int % 10 == 0:
        print(f"{alpha:>7.2f} {acc:>9.2f}% {correct:>6}/{n_test}")
    if acc > best_logit_acc:
        best_logit_acc = acc
        best_alpha_logit = alpha

print(f"\nBest logit fusion: alpha={best_alpha_logit:.2f} → {best_logit_acc:.2f}%")

# Per-class breakdown
print(f"\nPer-class oracle analysis:")
print(f"{'Class':>6} {'PMamba':>8} {'Quat':>8} {'Oracle':>8} {'Both':>6} {'Only_P':>7} {'Only_Q':>7} {'Neither':>8}")
for c in sorted(set(labels)):
    mask = labels == c
    n = mask.sum()
    p_acc = pmamba_correct[mask].sum()
    q_acc = quat_correct[mask].sum()
    o_acc = oracle_correct[mask].sum()
    b_acc = both_correct[mask].sum()
    op = only_pmamba[mask].sum()
    oq = only_quat[mask].sum()
    bw = both_wrong[mask].sum()
    print(f"{c:>6} {p_acc:>5}/{n:<3} {q_acc:>5}/{n:<3} {o_acc:>5}/{n:<3} {b_acc:>6} {op:>7} {oq:>7} {bw:>8}")
