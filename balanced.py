import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# CONFIGURATION
PROCESSED_DIR = r"D:\Deep Learning Course\Arrythmia Project\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0\processed"   # folder where preprocess.py saved the .npy files
OUTPUT_DIR    = r"D:\Deep Learning Course\Arrythmia Project\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0\processed"   # we'll overwrite with balanced versions

CLASS_NAMES = ['Normal', 'APC (SVE)', 'PVC (VE)', 'Fusion']


# STEP 1 — Load saved data
print("=" * 50)
print("Balancing Pipeline")
print("=" * 50)

X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

print(f"\n[1] Loaded — X_train: {X_train.shape}, X_test: {X_test.shape}")

# STEP 2 — Drop Class 4 (Unclassifiable)
train_mask = y_train != 4
test_mask  = y_test  != 4

X_train = X_train[train_mask]
y_train = y_train[train_mask]
X_test  = X_test[test_mask]
y_test  = y_test[test_mask]

print(f"\n[2] After dropping Class 4:")
print(f"    X_train: {X_train.shape}  X_test: {X_test.shape}")
print(f"    Train class counts: {dict(sorted(Counter(y_train).items()))}")
print(f"    Test  class counts: {dict(sorted(Counter(y_test).items()))}")

# STEP 3 — Oversample minority classes (train only)
# We use random oversampling (duplication with jitter)
# Target: bring all classes to at least 20% of Normal count
def oversample_with_jitter(X, y, target_counts, noise_std=0.005, seed=42):
    """
    Oversample minority classes to target_counts by duplicating samples
    and adding tiny Gaussian noise to avoid exact duplicates.
    """
    rng = np.random.default_rng(seed)
    X_out = [X]
    y_out = [y]

    for cls, target in target_counts.items():
        cls_idx = np.where(y == cls)[0]
        current = len(cls_idx)
        if current >= target:
            continue

        needed = target - current
        chosen = rng.choice(cls_idx, size=needed, replace=True)
        X_new  = X[chosen] + rng.normal(0, noise_std, (needed, X.shape[1], X.shape[2]))
        y_new  = np.full(needed, cls, dtype=y.dtype)

        X_out.append(X_new.astype(np.float32))
        y_out.append(y_new)

    X_balanced = np.concatenate(X_out, axis=0)
    y_balanced = np.concatenate(y_out, axis=0)

    # Shuffle
    idx = rng.permutation(len(y_balanced))
    return X_balanced[idx], y_balanced[idx]


# Compute targets: minority classes → 25% of Normal count
normal_count = int(np.sum(y_train == 0))
target_min   = int(normal_count * 0.25)   # ~18,800 each

counts = Counter(y_train)
target_counts = {
    cls: target_min
    for cls in [1, 2, 3]           # APC, PVC, Fusion
    if counts[cls] < target_min
}

print(f"\n[3] Oversampling minority classes to {target_min} each...")
X_train_bal, y_train_bal = oversample_with_jitter(X_train, y_train, target_counts)

print(f"    Balanced X_train: {X_train_bal.shape}")
print(f"    Balanced class counts: {dict(sorted(Counter(y_train_bal).items()))}")

# STEP 4 — Compute class weights (for use during training)
def compute_class_weights(y):
    counts  = Counter(y)
    total   = len(y)
    n_cls   = len(counts)
    weights = {cls: total / (n_cls * cnt) for cls, cnt in counts.items()}
    return weights

class_weights = compute_class_weights(y_train_bal)
print(f"\n[4] Class weights for training:")
for cls, w in sorted(class_weights.items()):
    print(f"    Class {cls} ({CLASS_NAMES[cls]}): {w:.4f}")

# STEP 5 — Save balanced data
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train_bal)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),  X_test)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train_bal)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),  y_test)
np.save(os.path.join(OUTPUT_DIR, "class_weights.npy"), class_weights)

print(f"\n[5] Saved balanced dataset to '{OUTPUT_DIR}/'")

# STEP 6 — Plot before vs after
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = ['steelblue', 'tomato', 'seagreen', 'darkorange']

for ax, labels, title in zip(
    axes,
    [y_train, y_train_bal],
    ['Before balancing (train)', 'After balancing (train)']
):
    counts_plot = Counter(labels)
    vals = [counts_plot.get(i, 0) for i in range(4)]
    bars = ax.bar(CLASS_NAMES, vals, color=colors, edgecolor='black')
    ax.set_title(title)
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=15)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 50, str(v),
                ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_balance_comparison.png"), dpi=150)
plt.show()
print("\nBalance comparison plot saved.")
print("\nDone. Ready for model training.")