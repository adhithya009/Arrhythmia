import os

import matplotlib
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import callbacks, layers, models

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_processed_dir():
    candidates = [
        os.environ.get("PROCESSED_DIR"),
        os.path.join(BASE_DIR, "processed"),
        os.path.join(
            BASE_DIR,
            "mit-bih-arrhythmia-database-1.0.0",
            "mit-bih-arrhythmia-database-1.0.0",
            "processed",
        ),
    ]

    for candidate in candidates:
        if candidate and os.path.exists(os.path.join(candidate, "X_train.npy")):
            return candidate

    raise FileNotFoundError(
        "Could not find processed data. Set PROCESSED_DIR or place the .npy files "
        "under ./processed or the nested dataset folder."
    )


PROCESSED_DIR = resolve_processed_dir()
OUTPUT_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["Normal", "APC", "PVC", "Fusion"]
NUM_CLASSES = 4
WINDOW_SIZE = 187
BATCH_SIZE = int(os.environ.get("TCN_BATCH_SIZE", "64"))
EPOCHS = int(os.environ.get("TCN_EPOCHS", "50"))
LEARNING_RATE = float(os.environ.get("TCN_LEARNING_RATE", "1e-3"))
EARLY_STOP_PATIENCE = int(os.environ.get("TCN_EARLY_STOP_PATIENCE", "12"))

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

print("=" * 50)
print("TCN Training Pipeline")
print("=" * 50)

X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
class_weights_raw = np.load(
    os.path.join(PROCESSED_DIR, "class_weights.npy"), allow_pickle=True
).item()
class_weights_dict = {int(key): float(value) for key, value in class_weights_raw.items()}

print(f"\n[1] Loaded - X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"    Using processed data from: {PROCESSED_DIR}")
print(f"    Class weights: {class_weights_dict}")


def residual_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.2):
    residual = x

    x = layers.Conv1D(
        filters,
        kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation=None,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv1D(
        filters,
        kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation=None,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    if residual.shape[-1] != filters:
        residual = layers.Conv1D(filters, 1, padding="same")(residual)

    return layers.Add()([x, residual])


def build_tcn(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv1D(32, kernel_size=1, padding="same")(inputs)

    for dilation in [1, 2, 4, 8]:
        x = residual_block(x, filters=32, kernel_size=5, dilation_rate=dilation)

    for dilation in [1, 2, 4]:
        x = residual_block(x, filters=64, kernel_size=3, dilation_rate=dilation)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="TCN")


model = build_tcn((WINDOW_SIZE, 1), NUM_CLASSES)
model.summary()

total_params = model.count_params()
size_kb = (total_params * 4) / 1024
print(f"\n    Estimated float32 model size: {size_kb:.1f} KB")
print(f"    Estimated INT8 model size:    {size_kb / 4:.1f} KB")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

best_model_path = os.path.join(OUTPUT_DIR, "best_tcn.keras")
cb_list = [
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    ),
    callbacks.ModelCheckpoint(
        best_model_path, monitor="val_loss", save_best_only=True, verbose=1
    ),
]

print("\n[2] Training...")
history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    class_weight=class_weights_dict,
    callbacks=cb_list,
    verbose=2,
)

model = tf.keras.models.load_model(best_model_path)

print("\n[3] Evaluating on test set...")
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
)
plt.title("TCN - Confusion Matrix (Test Set)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tcn_confusion_matrix.png"), dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["loss"], label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="Train Acc")
axes[1].plot(history.history["val_accuracy"], label="Val Acc")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tcn_training_curves.png"), dpi=150)
plt.close(fig)

print(f"\nModel saved to '{best_model_path}'")
print("Training complete.")
