import os
import numpy as np
import tensorflow as tf

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
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
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))
OUTPUT_DIR = os.environ.get("TFLITE_OUTPUT_DIR", os.path.join(BASE_DIR, "tflite"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES   = ['Normal', 'APC', 'PVC', 'Fusion']
REPRESENTATIVE_SAMPLES = 200   # samples used to calibrate INT8 ranges

# ─────────────────────────────────────────────
# STEP 1 — Load test data + representative set
# ─────────────────────────────────────────────
print("=" * 50)
print("INT8 Quantization Pipeline")
print("=" * 50)

X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

# Representative dataset for calibration — random sample from training set
rng = np.random.default_rng(42)
idx = rng.choice(len(X_train), size=REPRESENTATIVE_SAMPLES, replace=False)
representative_data = X_train[idx].astype(np.float32)

print(f"\n[1] Loaded data")
print(f"    X_test: {X_test.shape}")
print(f"    Representative calibration samples: {REPRESENTATIVE_SAMPLES}")


# ─────────────────────────────────────────────
# HELPER — Representative dataset generator
# Required by TFLite INT8 calibration
# ─────────────────────────────────────────────
def make_representative_dataset(data):
    def representative_dataset():
        for sample in data:
            yield [sample[np.newaxis, ...]]   # add batch dim
    return representative_dataset


# ─────────────────────────────────────────────
# HELPER — Quantize a model to INT8 TFLite
# ─────────────────────────────────────────────
def quantize_model(model_path, rep_data, output_path):
    print(f"\n    Loading {model_path}...")
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Full INT8 quantization — weights AND activations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset(rep_data)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"    Saved: {output_path}")
    print(f"    Actual INT8 size: {size_kb:.1f} KB")
    return tflite_model, size_kb


# ─────────────────────────────────────────────
# HELPER — Evaluate TFLite model on test set
# ─────────────────────────────────────────────
def evaluate_tflite(tflite_model, X_test, y_test, model_name):
    from sklearn.metrics import classification_report, accuracy_score

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale     = input_details['quantization'][0]
    input_zp        = input_details['quantization'][1]
    output_scale    = output_details['quantization'][0]
    output_zp       = output_details['quantization'][1]
    input_dtype     = input_details['dtype']

    predictions = []
    for sample in X_test:
        input_data = sample[np.newaxis, ...].astype(np.float32)

        if input_dtype == np.int8:
            if input_scale == 0:
                scaled = input_data.astype(np.int8)
            else:
                scaled = np.clip(
                    np.round(input_data / input_scale) + input_zp,
                    -128, 127
                ).astype(np.int8)
            interpreter.set_tensor(input_details['index'], scaled)
        else:
            interpreter.set_tensor(input_details['index'], input_data)

        interpreter.invoke()

        output = interpreter.get_tensor(output_details['index'])

        # Dequantize output if INT8
        if output_details['dtype'] == np.int8:
            output = (output.astype(np.float32) - output_zp) * output_scale

        predictions.append(np.argmax(output))

    y_pred = np.array(predictions)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n    {model_name} — Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

    return acc, y_pred


# ─────────────────────────────────────────────
# STEP 2 — Quantize CNN
# ─────────────────────────────────────────────
print("\n[2] Quantizing 1D-CNN...")
cnn_tflite, cnn_size = quantize_model(
    model_path  = os.path.join(MODELS_DIR, "best_cnn.keras"),
    rep_data    = representative_data,
    output_path = os.path.join(OUTPUT_DIR, "cnn_int8.tflite")
)

# ─────────────────────────────────────────────
# STEP 3 — Quantize TCN
# ─────────────────────────────────────────────
print("\n[3] Quantizing TCN...")
tcn_tflite, tcn_size = quantize_model(
    model_path  = os.path.join(MODELS_DIR, "best_tcn.keras"),
    rep_data    = representative_data,
    output_path = os.path.join(OUTPUT_DIR, "tcn_int8.tflite")
)

# ─────────────────────────────────────────────
# STEP 4 — Evaluate both INT8 models
# ─────────────────────────────────────────────
print("\n[4] Evaluating INT8 models on test set...")

print("\n  ── 1D-CNN INT8 ──")
cnn_acc, _ = evaluate_tflite(cnn_tflite, X_test, y_test, "1D-CNN INT8")

print("\n  ── TCN INT8 ──")
tcn_acc, _ = evaluate_tflite(tcn_tflite, X_test, y_test, "TCN INT8")

# ─────────────────────────────────────────────
# STEP 5 — Summary table
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("QUANTIZATION SUMMARY")
print("=" * 50)
print(f"{'Model':<12} {'Float32':>10} {'INT8':>10} {'Accuracy':>12}")
print("-" * 50)

# Reload float32 sizes
cnn_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "best_cnn.keras"))
tcn_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "best_tcn.keras"))
cnn_f32 = (cnn_model.count_params() * 4) / 1024
tcn_f32 = (tcn_model.count_params() * 4) / 1024

print(f"{'1D-CNN':<12} {cnn_f32:>8.1f}KB {cnn_size:>8.1f}KB {cnn_acc*100:>10.2f}%")
print(f"{'TCN':<12} {tcn_f32:>8.1f}KB {tcn_size:>8.1f}KB {tcn_acc*100:>10.2f}%")
print("=" * 50)
print("\nTFLite files saved to './tflite/'")
print("Next step: run convert_to_c_array.py to generate .h header files")
