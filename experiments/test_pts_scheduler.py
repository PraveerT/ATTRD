import matplotlib.pyplot as plt
import numpy as np

def pts_size_scheduler(epoch):
    """
    Dynamic pts_size scheduling:
    - Epoch 0-50: 96 -> 128 (slow linear increase)
    - Epoch 50-100: 128 -> 256 (fast quadratic increase)
    - Epoch 100+: 256 (constant)
    """
    if epoch < 50:
        # Slow linear increase from 96 to 128 over 50 epochs
        pts_size = int(96 + (128 - 96) * (epoch / 50))
    elif epoch < 100:
        # Fast exponential-like increase from 128 to 256 over 50 epochs
        progress = (epoch - 50) / 50  # 0 to 1
        # Use quadratic progression for faster increase
        pts_size = int(128 + (256 - 128) * (progress ** 2))
    else:
        # Keep at maximum after epoch 100
        pts_size = 256
    return pts_size

# Test the scheduler
epochs = list(range(150))
pts_sizes = [pts_size_scheduler(e) for e in epochs]

# Print some key epochs
print("PTS Size Scheduler Test:")
print("=" * 50)
for epoch in [0, 10, 25, 40, 49, 50, 60, 75, 90, 99, 100, 120]:
    print(f"Epoch {epoch:3d}: pts_size = {pts_size_scheduler(epoch):3d}")

print("\n" + "=" * 50)
print("Phase transitions:")
print(f"Phase 1 (0-50):   Start: {pts_size_scheduler(0)}, End: {pts_size_scheduler(49)}")
print(f"Phase 2 (50-100): Start: {pts_size_scheduler(50)}, End: {pts_size_scheduler(99)}")
print(f"Phase 3 (100+):   Constant: {pts_size_scheduler(100)}")

# Visualize the scheduler
plt.figure(figsize=(12, 6))
plt.plot(epochs, pts_sizes, linewidth=2)
plt.axvline(x=50, color='r', linestyle='--', alpha=0.5, label='Phase transition')
plt.axvline(x=100, color='r', linestyle='--', alpha=0.5)
plt.axhline(y=96, color='g', linestyle=':', alpha=0.3)
plt.axhline(y=128, color='g', linestyle=':', alpha=0.3)
plt.axhline(y=256, color='g', linestyle=':', alpha=0.3)
plt.xlabel('Epoch')
plt.ylabel('pts_size')
plt.title('Dynamic PTS Size Scheduling\n(Slow increase 96→128, then fast increase 128→256)')
plt.grid(True, alpha=0.3)
plt.xlim(0, 150)
plt.ylim(90, 260)

# Add annotations
plt.text(25, 100, 'Slow linear\nincrease', ha='center', fontsize=10, color='blue')
plt.text(75, 180, 'Fast quadratic\nincrease', ha='center', fontsize=10, color='blue')
plt.text(125, 250, 'Constant\n(max)', ha='center', fontsize=10, color='blue')

plt.tight_layout()
plt.savefig('pts_size_schedule.png', dpi=100)
print(f"\n📊 Visualization saved to pts_size_schedule.png")
plt.show()