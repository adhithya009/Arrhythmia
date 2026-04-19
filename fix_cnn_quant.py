import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score

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
        if candidate and os.path.exists(os.path.join(candidate, "X_test.npy")):
            return candidate

    raise FileNotFoundError(
        "Could not find processed data. Set PROCESSED_DIR or place the .npy files "
        "under ./processed or the nested dataset folder."
    )


PROCESSED_DIR = resolve_processed_dir()
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))
TFLITE_DIR = os.environ.get("TFLITE_DIR", os.path.join(BASE_DIR, "tflite"))
CLASS_NAMES   = ['Normal', 'APC', 'PVC', 'Fusion']

X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

print("=" * 50)
print("Diagnosing CNN INT8 Quantization")
print("=" * 50)

# ─────────────────────────────────────────────
# STEP 1 — Inspect quantization parameters
# ─────────────────────────────────────────────
def inspect_tflite(tflite_path):
    with open(tflite_path, 'rb') as f:
        tflite_model = f.read()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print(f"\n  Input  dtype:       {input_details['dtype']}")
    print(f"  Input  shape:       {input_details['shape']}")
    print(f"  Input  scale:       {input_details['quantization'][0]:.6f}")
    print(f"  Input  zero_point:  {input_details['quantization'][1]}")
    print(f"  Output dtype:       {output_details['dtype']}")
    print(f"  Output scale:       {output_details['quantization'][0]:.6f}")
    print(f"  Output zero_point:  {output_details['quantization'][1]}")
    return tflite_model, interpreter, input_details, output_details


print("\n[1] CNN INT8 parameters:")
cnn_tflite, cnn_interp, cnn_in, cnn_out = inspect_tflite(
    os.path.join(TFLITE_DIR, "cnn_int8.tflite")
)

print("\n[2] TCN INT8 parameters (for reference):")
tcn_tflite, tcn_interp, tcn_in, tcn_out = inspect_tflite(
    os.path.join(TFLITE_DIR, "tcn_int8.tflite")
)

# ─────────────────────────────────────────────
# STEP 2 — Robust evaluation
# Handles both INT8 and FLOAT32 input types
# ─────────────────────────────────────────────
def evaluate_tflite_robust(tflite_path, X_test, y_test, model_name):
    with open(tflite_path, 'rb') as f:
        tflite_model = f.read()

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
            # Proper INT8 scaling
            if input_scale == 0:
                # Fallback if scale is 0 — shouldn't happen but safety net
                scaled = input_data.astype(np.int8)
            else:
                scaled = np.clip(
                    np.round(input_data / input_scale) + input_zp,
                    -128, 127
                ).astype(np.int8)
            interpreter.set_tensor(input_details['index'], scaled)
        else:
            # Model expects float input despite INT8 weights
            interpreter.set_tensor(input_details['index'], input_data)

        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])

        # Dequantize output if INT8
        if output_details['dtype'] == np.int8:
            output = (output.astype(np.float32) - output_zp) * output_scale

        predictions.append(np.argmax(output))

    y_pred = np.array(predictions)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  {model_name} — Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))
    return acc


# ─────────────────────────────────────────────
# STEP 3 — Re-evaluate CNN with robust method
# ─────────────────────────────────────────────
print("\n[3] Re-evaluating CNN INT8 with robust scaling...")
cnn_acc = evaluate_tflite_robust(
    os.path.join(TFLITE_DIR, "cnn_int8.tflite"),
    X_test, y_test, "1D-CNN INT8"
)

# ─────────────────────────────────────────────
# STEP 4 — If still broken, requantize CNN with
# float input/output (hybrid quantization)
# ─────────────────────────────────────────────
if cnn_acc < 0.90:
    print("\n[4] CNN still degraded — requantizing with float I/O...")
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_train), size=500, replace=False)
    rep_data = X_train[idx].astype(np.float32)

    def representative_dataset():
        for sample in rep_data:
            yield [sample[np.newaxis, ...]]

    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "best_cnn.keras"))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Keep float I/O — INT8 weights/activations internally
    # This avoids the input scaling issue entirely
    # Note: float I/O has tiny overhead but negligible on ESP32

    tflite_fixed = converter.convert()

    fixed_path = os.path.join(TFLITE_DIR, "cnn_int8_fixed.tflite")
    with open(fixed_path, 'wb') as f:
        f.write(tflite_fixed)

    size_kb = len(tflite_fixed) / 1024
    print(f"  Saved fixed model: {fixed_path}")
    print(f"  Size: {size_kb:.1f} KB")

    print("\n  Evaluating fixed CNN...")
    cnn_acc_fixed = evaluate_tflite_robust(fixed_path, X_test, y_test, "1D-CNN INT8 (fixed)")

    if cnn_acc_fixed > 0.95:
        print("\n  Fixed model is good — renaming to cnn_int8.tflite")
        import shutil
        shutil.copy(fixed_path, os.path.join(TFLITE_DIR, "cnn_int8.tflite"))
        print("  Done.")
    else:
        print("\n  [!] Still degraded — may need quantization-aware training.")
        print("  Tell me the accuracy and we'll decide next steps.")

else:
    print(f"\n  CNN is fine with robust scaling ({cnn_acc*100:.2f}%)")
    print("  The original quantize.py just had a scaling bug for this model.")
