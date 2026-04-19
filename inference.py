#!/usr/bin/env python3
"""
Standalone Arrythmia Inference CLI
Fixed robust TFLite predictor for CNN/TCN models.

Usage:
  python inference.py --eval                    # Test set accuracy
  python inference.py --model cnn_qat_int8.tflite --beats X_test.npy
  python inference.py --beat single_beat.npy    # Single prediction
  python inference.py --raw ecg_samples.txt     # Raw ECG → predict

Fixed quantization handling from fix_cnn_quant.py.
"""
import os
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf

CLASS_NAMES = ['Normal', 'APC', 'PVC', 'Fusion']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get('MODELS_DIR', os.path.join(BASE_DIR, 'models'))
TFLITE_DIR = os.environ.get('TFLITE_DIR', os.path.join(BASE_DIR, 'tflite'))
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')  # fallback

def resolve_processed_dir():
    candidates = [
        os.environ.get('PROCESSED_DIR'),
        os.path.join(BASE_DIR, 'processed'),
        os.path.join(BASE_DIR, 'mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0/processed'),
    ]
    for path in candidates:
        if path and os.path.exists(os.path.join(path, 'X_test.npy')):
            return path
    raise FileNotFoundError('Processed data not found. Run preprocessing.py first.')

def load_tflite_model(model_path):
    if not os.path.exists(model_path):
        model_path = os.path.join(TFLITE_DIR, model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'TFLite model not found: {model_path}')
    with open(model_path, 'rb') as f:
        return f.read()

def robust_predict(tflite_model, X, model_name='Model'):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    input_scale, input_zp = input_details['quantization']
    output_scale, output_zp = output_details['quantization']
    input_dtype = input_details['dtype']
    
    predictions = []
    confidences = []
    
    for i, sample in enumerate(X):
        input_data = sample[np.newaxis, ...].astype(np.float32)
        
        if input_dtype == np.int8:
            if input_scale == 0:
                scaled = input_data.astype(np.int8)
            else:
                scaled = np.clip(np.round(input_data / input_scale) + input_zp, -128, 127).astype(np.int8)
            interpreter.set_tensor(input_details['index'], scaled)
        else:
            interpreter.set_tensor(input_details['index'], input_data)
        
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])
        
        if output_details['dtype'] == np.int8:
            output = (output.astype(np.float32) - output_zp) * output_scale
        
        pred = np.argmax(output)
        confidence = output[pred]
        predictions.append(pred)
        confidences.append(confidence)
        
        print(f'  Sample {i+1}: {CLASS_NAMES[pred]} ({confidence:.1%})')
    
    return np.array(predictions), np.array(confidences)

def preprocess_raw_ecg(raw_path, window_size=187):
    """Raw ECG line → normalized windowed beats."""
    data = np.loadtxt(raw_path, dtype=np.float32)
    data = (data - np.mean(data)) / (np.std(data) + 1e-8)  # z-score [-1,1]
    
    beats = []
    half = window_size // 2
    for i in range(half, len(data) - half):
        beat = data[i-half:i+half+1]
        beats.append(beat)
    
    return np.array(beats)

def main():
    parser = argparse.ArgumentParser(description='Arrythmia TFLite Inference')
    parser.add_argument('--model', default='cnn_qat_int8.tflite', help='TFLite model')
    parser.add_argument('--beats', help='Batch .npy file')
    parser.add_argument('--beat', help='Single beat .npy')
    parser.add_argument('--raw', help='Raw ECG .txt file')
    parser.add_argument('--eval', action='store_true', help='Eval on test set')
    parser.add_argument('--processed-dir', default=None, help='Processed data dir')
    args = parser.parse_args()
    
    if args.processed_dir:
        global PROCESSED_DIR
        PROCESSED_DIR = args.processed_dir
    
    model_data = load_tflite_model(args.model)
    print(f'Loaded {args.model}')
    
    if args.eval:
        processed_dir = resolve_processed_dir()
        X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
        
        y_pred, _ = robust_predict(model_data, X_test, args.model)
        acc = accuracy_score(y_test, y_pred)
        print(f'\n{args.model} Test Accuracy: {acc*100:.2f}%')
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    elif args.beats:
        X = np.load(args.beats)
        y_pred, conf = robust_predict(model_data, X[:,:,:,0] if X.ndim==4 else X, args.model)
        print(f'Batch pred shape: {y_pred.shape}')
    
    elif args.beat:
        X = np.load(args.beat)
        X = X[np.newaxis, ...] if X.ndim==2 else X
        y_pred, conf = robust_predict(model_data, X, args.model)
    
    elif args.raw:
        X = preprocess_raw_ecg(args.raw)
        y_pred, conf = robust_predict(model_data, X, args.model)
        print(f'Predicted {len(X)} windows from raw ECG')
    
    else:
        print('Use --eval, --beats, --beat, or --raw')
        return

if __name__ == '__main__':
    main()

