# Commit Checklist for Anemon Experiments

Workflow to prevent committing a number with a wrong mechanism story (the
quat-head 91.08 fiasco — turned out 71% of the lift was generic regularization,
not the named mechanism).

## Rule 0: Two-stage commit

Separate the *code-with-result* commit from the *mechanism-story* commit.

- **Stage 1 (code+number):** "X gives N% on NVGesture, mechanism unverified."
  Commit body lists what was changed and the empirical number. No causal claims.
- **Stage 2 (story):** "Verified mechanism: X's input/structure/etc accounts
  for K of the N% lift, the rest is generic regularization."
  Only after the control suite passes.

If the suite fails the story, Stage 2 never happens. The Stage-1 number still
stands; the framing doesn't.

## Rule 1: Control suite for "X helps" claims

Before claiming **why** a new module/feature helps, run all four. Each is a
fresh 100-epoch run with the same schedule.

| Control | What it tells you |
|---|---|
| **Ablate** — remove X entirely, retrain | Does the X-presence matter? Baseline gap = total lift. |
| **Freeze** — load trained ckpt, zero X's contribution at inference, re-eval | Is X inference-decorative? If acc unchanged, X is dead at eval. |
| **Zero-input** — replace X's input with constant zeros, keep params and scale | Does X's input content matter? If 91 still happens, the MLP structure regularizes regardless of input. |
| **Random-input** — replace X's input with `torch.randn`, keep everything else | Does X's input variation matter? If yes, the time-varying signal helps. |

If "X content matters" cannot be supported by *both* zero-input AND
random-input falling clearly below the real-X run, the mechanism claim is dead.

Cost on NVGesture: ~25-30 min per run × 4 = ~2 hours. Cheap vs reverting twice.

## Rule 2: Naming hygiene

File and class names should describe **structure**, not the **putative
mechanism**, until the mechanism is verified.

- ❌ `motion_cleanest_quat_head.py`, `MotionCleanestLinXLQuatHead`
- ✅ `motion_cleanest_auxhead.py`, `MotionCleanestLinXLAuxHead`

Rename only when controls confirm the mechanism does the named work.

## Rule 3: When skipping the suite is OK

- **Bug fix** (e.g., missing `.cuda()` call): smoke test only, commit allowed.
- **Speed optimization** (e.g., dataloader caching) that does not change loss
  math: smoke test + verify number is unchanged on at least one short run.
- **Dead code deletion**: smoke test, commit allowed.
- **Test-time tricks** (TTA, EMA) on an *already-controlled* trained model:
  fine to commit, but call out exactly what changed.

Everything else — new model variant, new aux head, new loss term, new
augmentation, new schedule — runs the suite first.

## Rule 4: Decomposing the lift

When the suite finishes, compute the breakdown as a table:

```
Variant                Best     Δ vs ablate
quat_head (real)       91.08    +1.45
zero_control           90.66    +1.03   ← 71% of lift from STRUCTURE
random_input           90.25    +0.62
ablate                 89.63       0
```

Mechanism claim is only credible if the gap **real vs (zero-input,
random-input both)** is at least as big as the gap **(zero-input or
random-input) vs ablate**. In the quat-head case it was not.

## When to invoke this

Anytime you're about to write a commit message that includes a phrase like:
- "X improves accuracy by N pp"
- "X is responsible for the lift"
- "X gives a real signal"
- "X beats the baseline because Y"

Run the suite first. Otherwise the next commit will need to be retracted.
