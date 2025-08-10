# 20BN-Jester Training Pipeline with Progressive Unfreezing

## Dataset Processing Status
- ✅ **50,420 training videos** extracted
- ✅ **6,981 test videos** extracted  
- 🔄 **Point cloud processing** in progress (57,401 total videos)

## Training Pipeline Commands

### Step 1: Complete Point Cloud Processing
```bash
cd /notebooks/PMamba/experiments
python jester_batch_process.py
```
*This processes all 57,401 videos with NVIDIA point cloud pipeline (RGB→grayscale→points)*

### Step 2: Pretraining on 20BN-Jester Dataset
```bash
cd /notebooks/PMamba/experiments
python main.py --config jester_pretrain.yaml --device 0
```

**Configuration:** `jester_pretrain.yaml`
- **Dataset:** 20BN-Jester (27 classes, 50K+ videos)
- **Architecture:** Motion + Mamba temporal encoder
- **Points:** 512 per frame, 32 frames
- **Epochs:** 100
- **LR:** 1e-4 with decay at [40, 60, 80]

### Step 3: Progressive Fine-tuning on NVGesture
```bash
cd /notebooks/PMamba/experiments
python main.py --config pointlstm_progressive_finetune.yaml --device 0 --weights work_dir/jester_pretrain/best_model.pt --fine-tuning=True
```

**Progressive Unfreezing Schedule:**
- **Epoch 0-2:** Only classifier (LR: 1e-5)
- **Epoch 3-12:** + Temporal encoder (LR: 8e-6)
- **Epoch 13-22:** + Late backbone (LR: 5e-6)
- **Epoch 23-32:** + Mid backbone (LR: 3e-6)
- **Epoch 33+:** + Early backbone (LR: 1e-6)

**LR Decay:** Epochs [20, 35, 45] → Ultra-safe rates (1e-7 to 1e-9)

### Step 4: Resume Training (if needed)
```bash
cd /notebooks/PMamba/experiments
python main.py --config pointlstm_progressive_finetune.yaml --device 0 --weights work_dir/nvgesture_progressive_finetuned/epochX_model.pt --resume=True
```

## Expected Results
- **Jester Pretraining:** Strong motion representation learning
- **NVGesture Fine-tuning:** 90%+ accuracy with no catastrophic forgetting
- **Ultra-safe LRs:** Prevent destruction of pretrained features

## Monitor Training
```bash
# Check training progress
tail -f work_dir/jester_pretrain/train.txt
tail -f work_dir/nvgesture_progressive_finetuned/train.txt

# Watch learning rates during progressive unfreezing
# Look for: [S1-2|1.0e-07 S3-4|3.0e-07 S5|5.0e-07 Temp|8.0e-07 S6|1.0e-06]
```

## Transfer Learning Strategy
1. **Large-scale pretraining** on Jester (50K videos, 27 classes)
2. **Progressive unfreezing** with discriminative learning rates
3. **Ultra-conservative LRs** to prevent catastrophic forgetting
4. **Multi-stage unfreezing** from classifier → temporal → backbone layers

## File Structure
```
work_dir/
├── jester_pretrain/          # Step 2 output
│   ├── best_model.pt         # Best pretrained model
│   └── train.txt            # Training logs
└── nvgesture_progressive_finetuned/  # Step 3 output
    ├── epochX_model.pt       # Progressive checkpoints  
    └── train.txt            # Fine-tuning logs
```