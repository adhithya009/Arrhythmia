# Arrhythmia

Personal deep-learning project that trains compact 1-D CNN and Temporal
Convolutional Network (TCN) models to classify single heartbeats from the
[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) and
deploys the quantized models to an ESP32 for on-wearable inference.

The pipeline covers four stages:

1. **Data prep** — extract `187`-sample beats around R-peaks from all 48
   MIT-BIH records, split/stratify, and oversample minority classes.
2. **Training** — float32 1-D CNN, TCN, and quantization-aware CNN (QAT).
3. **Post-training quantization** — export INT8 / float16 `.tflite` models.
4. **On-device** — convert `.tflite` to C headers and run TFLite-Micro on an
   ESP32 with live ADC input.

## Beat classes

Class 4 (unclassifiable/paced) is dropped during balancing; the deployed
models output four classes:

| ID | Label   | MIT-BIH symbols         |
|----|---------|-------------------------|
| 0  | Normal  | `N`, `.`, `n`           |
| 1  | APC     | `A`, `a`, `J`, `S`, `e`, `j` |
| 2  | PVC     | `V`, `E`                |
| 3  | Fusion  | `F`                     |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

TensorFlow 2.15 is pinned because `tensorflow_model_optimization` (used by
`cnn_qat.py`) only supports the `tf-keras` legacy API up to that version.

### Dataset

1. Download the MIT-BIH Arrhythmia Database 1.0.0 from PhysioNet.
2. Either extract it into `./mit-bih-arrhythmia-database-1.0.0/` at the repo
   root, or point the `PROCESSED_DIR` / scripts' `DATA_DIR` at your own
   location.

The training scripts look for the processed `.npy` files in this order:

1. `$PROCESSED_DIR` (if set)
2. `./processed/`
3. `./mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0/processed/`

## Pipeline

Run top-to-bottom. Each stage writes into directories that are gitignored.

```bash
# 1. Extract and window beats (edit DATA_DIR/OUTPUT_DIR at top of file)
python preprocessing.py

# 2. Drop class 4, oversample minorities, compute class weights
python balanced.py

# 3. Train a model (pick one or all)
python cnn.py            # 1-D CNN  -> models/best_cnn.keras
python tcn.py            # TCN      -> models/best_tcn.keras
python cnn_qat.py        # QAT CNN  -> models/cnn_qat_int8.tflite

# 4. Post-training quantization
python quantization.py       # INT8    -> tflite/*.tflite
python cnn_float16_quant.py  # Float16 -> tflite/cnn_float16.tflite

# 5. Package for ESP32
python c_array.py        # tflite/*.tflite  -> esp32/models/*_model.h
python test_beats.py     # holdout beats    -> esp32/arrhythmia_inference/test_samples.h

# 6. Evaluate a TFLite model from the CLI
python inference.py --eval --model cnn_qat_int8.tflite
```

### WSL helpers (optional)

The `run_*_wsl.sh` / `run_*_wsl.ps1` scripts are convenience launchers that
run the training entrypoints inside a WSL distro with a shared TensorFlow
env. They expect a `scripts/wsl_tensorflow_env.sh` helper to exist; create
your own activation script there (e.g. `source ~/venvs/tf/bin/activate`)
before using them.

## ESP32 firmware

Both `.ino` sketches live at the repo root for now; copy them into an
Arduino sketch folder alongside the generated `*_model.h` / `test_samples.h`
files before flashing.

- `esp32_arrhythmia_demo.ino` — live ECG on `A0` + test-sample benchmark,
  runs the CNN QAT INT8 model.
- `arrhythmia_inference_benchmark_v2.ino` — benchmark harness comparing
  CNN QAT INT8 and TCN INT8, with graduated arena fallback.

**Library:** `TensorFlowLite_ESP32` by tanakamasayuki (Arduino Library
Manager). See the header of `arrhythmia_inference_benchmark_v2.ino` for
notes on `EXPAND_DIMS` support and TCN arena sizing.

## Repo layout

```
preprocessing.py         data: MIT-BIH -> windowed beats (.npy)
balanced.py              data: drop class 4, oversample, class weights
cnn.py                   train: 1-D CNN
tcn.py                   train: TCN
cnn_qat.py               train: quantization-aware CNN (tf-keras + TFMOT)
quantization.py          deploy: post-training INT8 TFLite
cnn_float16_quant.py     deploy: post-training float16 TFLite
c_array.py               deploy: .tflite -> C header for TFLite-Micro
test_beats.py            deploy: MIT-BIH test beats -> C header
inference.py             eval/predict CLI for any TFLite model
run_*_wsl.sh / .ps1      WSL convenience launchers
*.ino                    ESP32 firmware
```

Generated artifacts (`processed/`, `models/`, `tflite/`, `esp32/models/`,
generated `*_model.h`, `test_samples.h`, plots) are gitignored and should
be rebuilt from the scripts above.
