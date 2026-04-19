import os

import matplotlib
import numpy as np
import keras as keras3
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras
from sklearn.metrics import accuracy_score, classification_report
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import (
    default_8bit_quantize_configs,
    default_8bit_quantize_registry,
    default_8bit_quantize_scheme,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = ["Normal", "APC", "PVC", "Fusion"]
BATCH_SIZE = int(os.environ.get("CNN_QAT_BATCH_SIZE", "32"))
EPOCHS = int(os.environ.get("CNN_QAT_EPOCHS", "8"))
LEARNING_RATE = float(os.environ.get("CNN_QAT_LEARNING_RATE", "1e-4"))
REPRESENTATIVE_SAMPLES = int(os.environ.get("CNN_QAT_REP_SAMPLES", "400"))
WINDOW_SIZE = 187
NUM_CLASSES = 4


class Custom8BitQuantizeRegistry(
    default_8bit_quantize_registry.Default8BitQuantizeRegistry
):
    """Extend the default 8-bit registry to cover this 1D CNN stack."""

    def __init__(self, disable_per_axis=False):
        super().__init__(disable_per_axis=disable_per_axis)

        # TFMOT's per-axis Conv quantizer is built for 2D kernels and trips on
        # Conv1D weights, so keep Conv1D on the stable per-tensor path here.
        self._layer_quantize_map[
            keras.layers.Conv1D
        ] = default_8bit_quantize_registry.Default8BitQuantizeConfig(
            ["kernel"], ["activation"], False
        )

        # These layers are fine to keep as pass-through during QAT.
        self._layer_quantize_map[
            keras.layers.MaxPooling1D
        ] = default_8bit_quantize_configs.NoOpQuantizeConfig()
        self._layer_quantize_map[
            keras.layers.BatchNormalization
        ] = default_8bit_quantize_configs.NoOpQuantizeConfig()


class Custom8BitQuantizeScheme(default_8bit_quantize_scheme.Default8BitQuantizeScheme):
    def get_quantize_registry(self):
        return Custom8BitQuantizeRegistry(disable_per_axis=self._disable_per_axis)


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


def build_legacy_cnn(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    x = keras.layers.Conv1D(32, kernel_size=5, padding="same", activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(32, kernel_size=5, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalAveragePooling1D()(x)

    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="1D_CNN_QAT")


def representative_dataset(data):
    for sample in data:
        yield [sample[np.newaxis, ...].astype(np.float32)]


def evaluate_tflite_model(tflite_model, x_test, y_test, model_name):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    predictions = []
    for sample in x_test:
        input_data = sample[np.newaxis, ...].astype(np.float32)

        if input_details["dtype"] == np.int8:
            quantized = np.clip(
                np.round(input_data / input_scale) + input_zero_point,
                -128,
                127,
            ).astype(np.int8)
            interpreter.set_tensor(input_details["index"], quantized)
        else:
            interpreter.set_tensor(input_details["index"], input_data)

        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])

        if output_details["dtype"] == np.int8:
            output = (output.astype(np.float32) - output_zero_point) * output_scale

        predictions.append(np.argmax(output))

    y_pred = np.array(predictions)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{model_name} accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))
    return accuracy


processed_dir = resolve_processed_dir()
models_dir = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))
tflite_dir = os.environ.get("TFLITE_DIR", os.path.join(BASE_DIR, "tflite"))
os.makedirs(tflite_dir, exist_ok=True)

base_model_path = os.path.join(models_dir, "best_cnn.keras")
qat_weights_path = os.path.join(models_dir, "best_cnn_qat.weights.h5")
qat_tflite_path = os.path.join(tflite_dir, "cnn_qat_int8.tflite")
history_plot_path = os.path.join(models_dir, "cnn_qat_training_curves.png")

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

print("=" * 50)
print("CNN Quantization-Aware Training")
print("=" * 50)

x_train = np.load(os.path.join(processed_dir, "X_train.npy"))
x_test = np.load(os.path.join(processed_dir, "X_test.npy"))
y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
class_weights_raw = np.load(
    os.path.join(processed_dir, "class_weights.npy"), allow_pickle=True
).item()
class_weights = {int(key): float(value) for key, value in class_weights_raw.items()}

rng = np.random.default_rng(42)
rep_indices = rng.choice(len(x_train), size=REPRESENTATIVE_SAMPLES, replace=False)
rep_data = x_train[rep_indices]

print(f"\nLoaded data from: {processed_dir}")
print(f"QAT epochs: {EPOCHS}, batch size: {BATCH_SIZE}, learning rate: {LEARNING_RATE}")

original_model = keras3.models.load_model(base_model_path)
legacy_model = build_legacy_cnn((WINDOW_SIZE, 1), NUM_CLASSES)
legacy_model.set_weights(original_model.get_weights())
annotated_model = tfmot.quantization.keras.quantize_annotate_model(legacy_model)

with tfmot.quantization.keras.quantize_scope(
    {
        "Custom8BitQuantizeRegistry": Custom8BitQuantizeRegistry,
        "Custom8BitQuantizeScheme": Custom8BitQuantizeScheme,
    }
):
    qat_model = tfmot.quantization.keras.quantize_apply(
        annotated_model,
        scheme=Custom8BitQuantizeScheme(),
    )
qat_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

cb_list = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        qat_weights_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    ),
]

print("\nStarting QAT fine-tuning...")
history = qat_model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    class_weight=class_weights,
    callbacks=cb_list,
    verbose=2,
)

qat_model.load_weights(qat_weights_path)

converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = lambda: representative_dataset(rep_data)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

qat_tflite_model = converter.convert()

with open(qat_tflite_path, "wb") as file_obj:
    file_obj.write(qat_tflite_model)

float32_size_kb = (original_model.count_params() * 4) / 1024
int8_size_kb = len(qat_tflite_model) / 1024

print(f"\nSaved QAT weights to: {qat_weights_path}")
print(f"Saved QAT INT8 TFLite model to: {qat_tflite_path}")
print(f"Base float32 size estimate: {float32_size_kb:.1f} KB")
print(f"QAT INT8 size:              {int8_size_kb:.1f} KB")

accuracy = evaluate_tflite_model(qat_tflite_model, x_test, y_test, "CNN QAT INT8")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["loss"], label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[0].set_title("QAT Loss")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="Train Acc")
axes[1].plot(history.history["val_accuracy"], label="Val Acc")
axes[1].set_title("QAT Accuracy")
axes[1].legend()

plt.tight_layout()
plt.savefig(history_plot_path, dpi=150)
plt.close(fig)

print(f"\nSaved QAT training curves to: {history_plot_path}")
print(f"Final QAT INT8 accuracy: {accuracy * 100:.2f}%")
