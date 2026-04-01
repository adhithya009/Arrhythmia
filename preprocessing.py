import os
import numpy as np
import wfdb
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


DATA_DIR = r"D:\Deep Learning Course\Arrythmia Project\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0"          
OUTPUT_DIR = r"D:\Deep Learning Course\Arrythmia Project\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0\processed"      
WINDOW_SIZE = 187               # samples per beat (standard for MIT-BIH at 360Hz)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# BEAT LABEL MAPPING  

# We collapse -50 annotation symbols into 5 clinically meaningful classes
LABEL_MAP = {
    # Class 0 — Normal
    'N': 0, '.': 0, 'n': 0,
    # Class 1 — Supraventricular ectopic (APC)
    'A': 1, 'a': 1, 'J': 1, 'S': 1, 'e': 1, 'j': 1,
    # Class 2 — Ventricular ectopic (PVC)
    'V': 2, 'E': 2,
    # Class 3 — Fusion beats
    'F': 3,
    # Class 4 — Unclassifiable / paced
    '/': 4, 'f': 4, 'Q': 4, 'q': 4,
}

# All 48 MIT-BIH records
ALL_RECORDS = [
    '100','101','102','103','104','105','106','107',
    '108','109','111','112','113','114','115','116',
    '117','118','119','121','122','123','124','200',
    '201','202','203','205','207','208','209','210',
    '212','213','214','215','217','219','220','221',
    '222','223','228','230','231','232','233','234'
]

# STEP 1 — Extract beats from all records
def extract_beats(data_dir, records, window_size):
    all_beats = []
    all_labels = []
    skipped = 0

    for rec_id in records:
        rec_path = os.path.join(data_dir, rec_id)
        try:
            # Read signal (use channel 0 = MLII lead)
            record = wfdb.rdrecord(rec_path, channels=[0])
            annotation = wfdb.rdann(rec_path, 'atr')
            signal = record.p_signal[:, 0]

            # Normalize signal to [-1, 1] per record
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

            r_peaks = annotation.sample
            symbols = annotation.symbol

            half = window_size // 2

            for peak, sym in zip(r_peaks, symbols):
                if sym not in LABEL_MAP:
                    skipped += 1
                    continue

                start = peak - half
                end = peak + (window_size - half)

                # Skip beats too close to signal edges
                if start < 0 or end > len(signal):
                    skipped += 1
                    continue

                beat = signal[start:end]
                all_beats.append(beat)
                all_labels.append(LABEL_MAP[sym])

        except Exception as e:
            print(f"  [!] Skipped record {rec_id}: {e}")

    print(f"\nExtracted {len(all_beats)} beats | Skipped {skipped} beats")
    return np.array(all_beats, dtype=np.float32), np.array(all_labels, dtype=np.int32)


# STEP 2 — Split and save
def save_dataset(beats, labels, output_dir):
    X_train, X_test, y_train, y_test = train_test_split(
        beats, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # Reshape for 1D-CNN input: (samples, timesteps, channels)
    X_train = X_train[..., np.newaxis]
    X_test  = X_test[...,  np.newaxis]

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"),  y_test)

    print(f"\nSaved to '{output_dir}/'")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")
    return X_train, X_test, y_train, y_test


# STEP 3 — Visualize class distribution
CLASS_NAMES = ['Normal', 'APC (SVE)', 'PVC (VE)', 'Fusion', 'Unclassifiable']

def plot_distribution(y_train, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, labels, title in zip(axes, [y_train, y_test], ['Train', 'Test']):
        counts = Counter(labels)
        names  = [CLASS_NAMES[i] for i in sorted(counts)]
        values = [counts[i] for i in sorted(counts)]
        ax.bar(names, values, color='steelblue', edgecolor='black')
        ax.set_title(f'{title} set — {len(labels)} beats')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=15)
        for i, v in enumerate(values):
            ax.text(i, v + 20, str(v), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=150)
    plt.show()
    print("Distribution plot saved.")


# STEP 4 — Visualize sample beats
def plot_sample_beats(X_train, y_train):
    fig, axes = plt.subplots(1, 5, figsize=(16, 3))
    for cls in range(5):
        idx = np.where(y_train == cls)[0]
        if len(idx) == 0:
            axes[cls].set_title(f'{CLASS_NAMES[cls]}\n(none)')
            continue
        beat = X_train[idx[0], :, 0]
        axes[cls].plot(beat, linewidth=1.2)
        axes[cls].set_title(CLASS_NAMES[cls], fontsize=9)
        axes[cls].set_xlabel("Sample")
        axes[cls].set_xticks([])
    plt.suptitle("One sample beat per class", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_beats.png"), dpi=150)
    plt.show()
    print("Sample beats plot saved.")


# MAIN
if __name__ == "__main__":
    print("=" * 50)
    print("MIT-BIH Preprocessing Pipeline")
    print("=" * 50)

    print(f"\n[1] Extracting beats from {len(ALL_RECORDS)} records...")
    beats, labels = extract_beats(DATA_DIR, ALL_RECORDS, WINDOW_SIZE)

    print("\n[2] Class counts (full dataset):")
    for cls, name in enumerate(CLASS_NAMES):
        count = np.sum(labels == cls)
        print(f"    Class {cls} ({name}): {count}")

    print("\n[3] Splitting and saving...")
    X_train, X_test, y_train, y_test = save_dataset(beats, labels, OUTPUT_DIR)

    print("\n[4] Plotting class distribution...")
    plot_distribution(y_train, y_test)

    print("\n[5] Plotting sample beats...")
    plot_sample_beats(X_train, y_train)

    print("\nPreprocessing complete.")