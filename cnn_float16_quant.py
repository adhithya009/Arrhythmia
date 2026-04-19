import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = ["Normal", "APC", "PVC", "Fusion"]


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


def evaluate_tflite_model(tflite_model, x_test, y_test, model_name):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = []
    for sample in x_test:
        input_data = sample[np.newaxis, ...].astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])

        if np.issubdtype(output_details["dtype"], np.integer):
            output_scale, output_zero_point = output_details["quantization"]
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

model_path = os.path.join(models_dir, "best_cnn.keras")
output_path = os.path.join(tflite_dir, "cnn_float16.tflite")

print("=" * 50)
print("CNN Float16 Quantization")
print("=" * 50)

x_test = np.load(os.path.join(processed_dir, "X_test.npy"))
y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

model = tf.keras.models.load_model(model_path)
float32_size_kb = (model.count_params() * 4) / 1024

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open(output_path, "wb") as file_obj:
    file_obj.write(tflite_model)

float16_size_kb = len(tflite_model) / 1024

print(f"\nSaved float16 TFLite model to: {output_path}")
print(f"Float32 size estimate: {float32_size_kb:.1f} KB")
print(f"Float16 TFLite size:   {float16_size_kb:.1f} KB")

evaluate_tflite_model(tflite_model, x_test, y_test, "CNN Float16 TFLite")
