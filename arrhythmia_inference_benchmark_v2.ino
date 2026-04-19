/**
 * arrhythmia_inference_benchmark_v2.ino
 *
 * ── What changed from v1 ────────────────────────────────────────────────────
 *  v1 problem 1: MicroMutableOpResolver was missing EXPAND_DIMS, which the
 *                CNN QAT model graph also contains → AllocateTensors FAILED.
 *  v1 problem 2: TCN arena (140 KB) could not be allocated from 349 KB heap
 *                because the internal DRAM has no single 140 KB contiguous block.
 *
 *  FIX 1: Replace MicroMutableOpResolver<N> with AllOpsResolver.
 *          AllOpsResolver registers every built-in op, including EXPAND_DIMS,
 *          and was the resolver used in the first working benchmark run.
 *          No manual AddX() calls are needed — remove those entirely.
 *
 *  FIX 2: RUN_TCN_MODEL defaults to 0.  TCN requires EXPAND_DIMS *kernel*
 *          support in the library build (separate from resolver registration),
 *          and the library prints "runtime hangs" for this board.
 *          Set RUN_TCN_MODEL to 1 only after either:
 *            (a) upgrading to espressif/esp-tflite-micro, or
 *            (b) re-exporting the TCN without EXPAND_DIMS nodes.
 *
 *  FIX 3: Graduated TCN arena fallback: tries 128 → 112 → 96 → 80 → 64 KB
 *          until one fits in available contiguous DRAM.
 *
 *  FIX 4: AllocateTensors() called exactly once per model.
 *          init_runner() uses placement-new so each interpreter is freshly
 *          constructed and the allocator state is clean.
 *
 * ── Required files in sketch folder ─────────────────────────────────────────
 *  tcn_int8_model.h
 *  cnn_qat_int8_model.h
 *  test_samples.h
 *
 * ── Library ──────────────────────────────────────────────────────────────────
 *  TensorFlowLite_ESP32 by tanakamasayuki  (Arduino Library Manager)
 * ─────────────────────────────────────────────────────────────────────────────
 */

// ── Model + sample data headers ──────────────────────────────────────────────
#include "tcn_int8_model.h"
#include "cnn_qat_int8_model.h"
#include "test_samples.h"

#include <math.h>
#include <new>

// ── TFLite Micro runtime ─────────────────────────────────────────────────────
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"   // ← KEY FIX: use AllOpsResolver
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_heap_caps.h"

// version.h is not shipped by all Arduino TFLM builds; define the constant manually
#ifndef TFLITE_SCHEMA_VERSION
#define TFLITE_SCHEMA_VERSION 3
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Tunable constants
// ─────────────────────────────────────────────────────────────────────────────
#define N_INPUTS        187
#define N_OUTPUTS       4
#define N_WARMUP_RUNS   5    // discarded passes (cache warm-up, not timed)
#define N_BENCH_RUNS    20   // timed passes — average is reported

//
// Set RUN_TCN_MODEL to 1 ONLY after:
//   (a) switching to espressif/esp-tflite-micro library, OR
//   (b) re-exporting the TCN model without EXPAND_DIMS nodes.
// Reason: this library build does not have the EXPAND_DIMS kernel registered
// even in AllOpsResolver, causing AllocateTensors to fail for TCN.
//
#define RUN_TCN_MODEL   0

// CNN QAT INT8 uses ~18 KB arena. 28 KB gives comfortable headroom.
#define CNN_QAT_ARENA_SIZE  (28 * 1024)

// TCN arena: graduated fallback — tries each size until one fits in DRAM
static const size_t TCN_ARENA_SIZES[] = {
  128 * 1024,
  112 * 1024,
  96  * 1024,
  80  * 1024,
  64  * 1024
};
static const int TCN_ARENA_STEPS = sizeof(TCN_ARENA_SIZES) / sizeof(TCN_ARENA_SIZES[0]);

// ─────────────────────────────────────────────────────────────────────────────
// Globals
// ─────────────────────────────────────────────────────────────────────────────
const char* CLASS_NAMES[N_OUTPUTS] = {"Normal", "APC", "PVC", "Fusion"};

static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter*     error_reporter = &micro_error_reporter;

// AllOpsResolver is stateless (a pure op-lookup table) — safe to share.
// It registers every built-in op including EXPAND_DIMS, DEQUANTIZE, etc.
static tflite::AllOpsResolver resolver;

// ── Model runner struct ───────────────────────────────────────────────────────
struct ModelRunner {
  const char*               name;
  const unsigned char*      model_data;
  const tflite::Model*      model;
  tflite::MicroInterpreter* interpreter;
  TfLiteTensor*             input;
  TfLiteTensor*             output;
  uint8_t*                  arena;
  size_t                    arena_size;
  size_t                    arena_used;
  bool                      ready;
};

static ModelRunner tcn_runner = {
  "TCN INT8", tcn_int8_model_data,
  nullptr, nullptr, nullptr, nullptr,
  nullptr, 0, 0, false
};

static ModelRunner cnn_qat_runner = {
  "CNN QAT INT8", cnn_qat_int8_model_data,
  nullptr, nullptr, nullptr, nullptr,
  nullptr, CNN_QAT_ARENA_SIZE, 0, false
};

// Placement-new storage — one buffer per interpreter.
// Placement-new lets us call the constructor on a stack/static buffer so
// the interpreter object itself doesn't fragment the heap.
alignas(16) static uint8_t tcn_interp_buf [sizeof(tflite::MicroInterpreter)];
alignas(16) static uint8_t cnn_interp_buf [sizeof(tflite::MicroInterpreter)];

// ─────────────────────────────────────────────────────────────────────────────
// Forward declarations
// ─────────────────────────────────────────────────────────────────────────────
uint8_t* alloc_arena (size_t bytes);
bool     init_runner (ModelRunner& r, uint8_t* ibuf);
bool     run_model   (ModelRunner& r, const float* sample, int true_label);
void     fill_input  (TfLiteTensor* t, const float* data, int len);
void     read_output (TfLiteTensor* t, float* out, int len);
int      argmax      (const float* a, int len);

// ─────────────────────────────────────────────────────────────────────────────
// Arena allocator: PSRAM first, then DRAM
// ─────────────────────────────────────────────────────────────────────────────
uint8_t* alloc_arena(size_t bytes) {
  uint8_t* p = (uint8_t*)heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!p)   p = (uint8_t*)heap_caps_malloc(bytes, MALLOC_CAP_8BIT);
  return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// init_runner — constructs a fresh interpreter, allocates tensors exactly once.
// Returns true on success; on failure sets r.ready=false and returns false.
// Never call AllocateTensors() again on the same runner — call this once only.
// ─────────────────────────────────────────────────────────────────────────────
bool init_runner(ModelRunner& r, uint8_t* ibuf) {
  r.ready       = false;
  r.model       = nullptr;
  r.interpreter = nullptr;
  r.input       = nullptr;
  r.output      = nullptr;
  r.arena_used  = 0;

  // 1. Parse model flatbuffer from flash
  r.model = tflite::GetModel(r.model_data);
  if (r.model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[%s] Schema mismatch: model=%d  runtime=%d\n",
                  r.name, (int)r.model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  // 2. Construct interpreter via placement-new.
  //    This ensures a clean allocator state every time init_runner is called.
  auto* interp = new (ibuf) tflite::MicroInterpreter(
      r.model, resolver, r.arena, r.arena_size, error_reporter);
  r.interpreter = interp;

  // 3. AllocateTensors — called ONCE, here, never inside the sample loop
  if (r.interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.printf("[%s] AllocateTensors FAILED\n", r.name);
    r.interpreter = nullptr;
    return false;
  }

  // 4. Cache input/output tensor pointers
  r.input  = r.interpreter->input(0);
  r.output = r.interpreter->output(0);
  if (!r.input || !r.output) {
    Serial.printf("[%s] Null tensor pointer after allocation\n", r.name);
    r.interpreter = nullptr;
    return false;
  }

  r.arena_used = r.interpreter->arena_used_bytes();
  r.ready = true;

  // Report RAM usage — this is your paper's embedded RAM metric
  Serial.printf("[%s] Ready  Arena: %u / %u bytes\n",
                r.name, (unsigned)r.arena_used, (unsigned)r.arena_size);
  Serial.printf("[%s]        Input : type=%d  bytes=%d\n",
                r.name, r.input->type,  r.input->bytes);
  Serial.printf("[%s]        Output: type=%d  bytes=%d\n",
                r.name, r.output->type, r.output->bytes);
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// setup — init models once, then run benchmark loop
// ─────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(3000);

  Serial.println("======================================================");
  Serial.println(" Arrhythmia Detection on ESP32 - Benchmark v2");
  Serial.println(" TCN INT8  |  CNN QAT INT8");
  Serial.println("======================================================");
  Serial.printf("Free heap  : %u bytes\n",   ESP.getFreeHeap());
  Serial.printf("Flash size : %u bytes\n\n", ESP.getFlashChipSize());

  // ── Init CNN QAT INT8 (primary model) ──────────────────────────────────────
  cnn_qat_runner.arena = alloc_arena(CNN_QAT_ARENA_SIZE);
  if (!cnn_qat_runner.arena) {
    Serial.printf("[CNN QAT INT8] Arena alloc failed (%u bytes requested)\n",
                  (unsigned)CNN_QAT_ARENA_SIZE);
  } else {
    init_runner(cnn_qat_runner, cnn_interp_buf);
  }

  // ── Init TCN INT8 (optional) ───────────────────────────────────────────────
#if RUN_TCN_MODEL
  // Graduated arena fallback: finds the largest contiguous block available
  for (int i = 0; i < TCN_ARENA_STEPS && !tcn_runner.arena; i++) {
    size_t sz = TCN_ARENA_SIZES[i];
    tcn_runner.arena = alloc_arena(sz);
    if (tcn_runner.arena) {
      tcn_runner.arena_size = sz;
      Serial.printf("[TCN INT8] Arena allocated: %u bytes\n", (unsigned)sz);
    } else {
      Serial.printf("[TCN INT8] %u bytes unavailable, trying smaller...\n", (unsigned)sz);
    }
  }
  if (!tcn_runner.arena) {
    Serial.println("[TCN INT8] All arena sizes failed — not enough contiguous DRAM.");
  } else {
    if (!init_runner(tcn_runner, tcn_interp_buf)) {
      Serial.println("[TCN INT8] Init failed.");
      Serial.println("           If error says EXPAND_DIMS: re-export model or upgrade");
      Serial.println("           to espressif/esp-tflite-micro for full op support.");
    }
  }
#else
  Serial.println("[TCN INT8] Skipped (RUN_TCN_MODEL=0)");
  Serial.println("           Enable after fixing EXPAND_DIMS op support for this model.");
#endif

  Serial.println();
  Serial.println("======================================================");

  // ── Benchmark sample loop ──────────────────────────────────────────────────
  // Only Invoke() happens here — no AllocateTensors(), no model loading.
  for (int s = 0; s < NUM_TEST_SAMPLES; s++) {
    const float* sample = test_samples[s];
    int          label  = test_labels[s];

    Serial.printf("\n--- Sample %d/%d  |  True class: %s ---\n",
                  s + 1, NUM_TEST_SAMPLES, CLASS_NAMES[label]);

#if RUN_TCN_MODEL
    if (tcn_runner.ready)
      run_model(tcn_runner, sample, label);
    else
      Serial.println("  [TCN INT8     ] Skipped");
#else
    Serial.println("  [TCN INT8     ] Skipped (RUN_TCN_MODEL=0)");
#endif

    if (cnn_qat_runner.ready)
      run_model(cnn_qat_runner, sample, label);
    else
      Serial.println("  [CNN QAT INT8 ] Skipped (init failed — check serial log above)");

    delay(50);
  }

  Serial.println();
  Serial.println("=======================================");
  Serial.println("  Benchmark complete.  Halting.");
  Serial.println("=======================================");
}

void loop() { /* intentionally empty */ }

// ─────────────────────────────────────────────────────────────────────────────
// run_model — warmup, timed benchmark, final readout
// AllocateTensors is NOT called here — only Invoke().
// ─────────────────────────────────────────────────────────────────────────────
bool run_model(ModelRunner& r, const float* sample, int true_label) {
  if (!r.ready || !r.interpreter) return false;

  // Warmup passes — discarded, not timed
  for (int i = 0; i < N_WARMUP_RUNS; i++) {
    fill_input(r.input, sample, N_INPUTS);
    if (r.interpreter->Invoke() != kTfLiteOk) {
      Serial.printf("  [%-12s] Invoke failed (warmup)\n", r.name);
      return false;
    }
  }

  // Timed passes
  unsigned long t0 = micros();
  for (int i = 0; i < N_BENCH_RUNS; i++) {
    fill_input(r.input, sample, N_INPUTS);
    r.interpreter->Invoke();   // error checked below in final readout
  }
  unsigned long avg_us = (micros() - t0) / N_BENCH_RUNS;

  // Final invoke for output readout
  fill_input(r.input, sample, N_INPUTS);
  if (r.interpreter->Invoke() != kTfLiteOk) {
    Serial.printf("  [%-12s] Invoke failed (readout)\n", r.name);
    return false;
  }

  float scores[N_OUTPUTS] = {};
  read_output(r.output, scores, N_OUTPUTS);
  int pred = argmax(scores, N_OUTPUTS);

  Serial.printf("  [%-12s] Pred: %-7s | %5lu us/inf | %s\n",
                r.name,
                CLASS_NAMES[pred],
                avg_us,
                (pred == true_label) ? "OK CORRECT" : "X WRONG");
  Serial.printf("               Scores: N=%.3f A=%.3f P=%.3f F=%.3f\n",
                scores[0], scores[1], scores[2], scores[3]);
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// fill_input — writes float[] into tensor, handling float32, int8, uint8
// ─────────────────────────────────────────────────────────────────────────────
void fill_input(TfLiteTensor* t, const float* data, int len) {
  if (!t) return;
  if (t->type == kTfLiteFloat32) {
    for (int i = 0; i < len; i++) t->data.f[i] = data[i];

  } else if (t->type == kTfLiteInt8) {
    const float   s  = t->params.scale;
    const int32_t zp = t->params.zero_point;
    for (int i = 0; i < len; i++) {
      int32_t q = (int32_t)roundf(data[i] / s) + zp;
      t->data.int8[i] = (int8_t)(q < -128 ? -128 : q > 127 ? 127 : q);
    }

  } else if (t->type == kTfLiteUInt8) {
    const float   s  = t->params.scale;
    const int32_t zp = t->params.zero_point;
    for (int i = 0; i < len; i++) {
      int32_t q = (int32_t)roundf(data[i] / s) + zp;
      t->data.uint8[i] = (uint8_t)(q < 0 ? 0 : q > 255 ? 255 : q);
    }
  }
  // Note: float16 input is not handled here because TFLite Micro's micro
  // dequantize kernel does not support float16 on this target. Do not attempt
  // to benchmark a float16 model on ESP32 with this library build.
}

// ─────────────────────────────────────────────────────────────────────────────
// read_output — reads output tensor into float[], handling float32, int8, uint8
// ─────────────────────────────────────────────────────────────────────────────
void read_output(TfLiteTensor* t, float* out, int len) {
  if (!t || !out) return;
  if (t->type == kTfLiteFloat32) {
    for (int i = 0; i < len; i++) out[i] = t->data.f[i];

  } else if (t->type == kTfLiteInt8) {
    const float   s  = t->params.scale;
    const int32_t zp = t->params.zero_point;
    for (int i = 0; i < len; i++) out[i] = (t->data.int8[i] - zp) * s;

  } else if (t->type == kTfLiteUInt8) {
    const float   s  = t->params.scale;
    const int32_t zp = t->params.zero_point;
    for (int i = 0; i < len; i++) out[i] = (t->data.uint8[i] - zp) * s;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// argmax
// ─────────────────────────────────────────────────────────────────────────────
int argmax(const float* a, int len) {
  int best = 0;
  for (int i = 1; i < len; i++) if (a[i] > a[best]) best = i;
  return best;
}
