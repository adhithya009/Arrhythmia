/**
 * Fixed ESP32 Arrythmia Demo - Live ECG + Test Benchmarks
 * FIXED: Live ADC input, proper scaling, confidence %, test acc summary.
 * 
 * Connect ECG sensor to A0 (3.3V ADC).
 * Serial @115200 for predictions + stats.
 */

#include "cnn_qat_int8_model.h"
#include "test_samples.h"
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <esp_heap_caps.h>
#include <math.h>

#define N_INPUTS 187
#define N_OUTPUTS 4
#define ECG_BUFFER_SIZE N_INPUTS
#define MIN_CONFIDENCE 0.7f

static uint8_t tensor_arena[200 * 1024];
static tflite::MicroErrorReporter micro_error_reporter;
static tflite::AllOpsResolver resolver;
const char* CLASS_NAMES[N_OUTPUTS] = {{"Normal", "APC", "PVC", "Fusion"}};

float ecg_buffer[ECG_BUFFER_SIZE];
int buffer_pos = 0;
bool buffer_full = false;

// Interpreter state
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  delay(2000);

  Serial.println("\n=== FIXED Arrythmia ESP32 Demo ===");
  Serial.println("Live ECG on A0 | Test Benchmarks | Robust INT8");

  // Allocate interpreter
  model = tflite::GetModel(cnn_qat_int8_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Schema mismatch!");
    return;
  }

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, sizeof(tensor_arena), &micro_error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.printf("Alloc fail! Used: %d/%d bytes\n", interpreter->arena_used_bytes(), sizeof(tensor_arena));
    return;
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
  
  Serial.printf("Model loaded. Arena used: %d bytes\n", interpreter->arena_used_bytes());
  Serial.println("Ready for ECG on A0...\n");

  // Run test benchmarks first
  run_test_benchmarks();
}

void loop() {
  // Live ECG prediction
  int adc_raw = analogRead(A0);  // 0-4095
  float voltage = (adc_raw / 4095.0) * 3.3;
  
  // Normalize to training range [-1.0, 1.0] assuming 1.65V baseline
  float normalized = (voltage - 1.65) / 1.65;  // [-1,1]
  normalized = max(-1.0f, min(1.0f, normalized));  // clip
  
  ecg_buffer[buffer_pos] = normalized;
  buffer_pos = (buffer_pos + 1) % ECG_BUFFER_SIZE;
  if (buffer_pos == 0) buffer_full = true;

  if (buffer_full) {
    predict_live();
  }
  
  delay(3);  // ~333Hz = 360Hz target
}

void predict_live() {
  // Copy buffer to input (float or int8 auto-handled)
  fill_input(input_tensor, ecg_buffer, N_INPUTS);
  interpreter->Invoke();
  
  float scores[N_OUTPUTS];
  read_output(output_tensor, scores, N_OUTPUTS);
  float probs[N_OUTPUTS];
  softmax(scores, probs, N_OUTPUTS);
  
  int pred = argmax(probs, N_OUTPUTS);
  float conf = probs[pred];
  
  if (conf > MIN_CONFIDENCE) {
    Serial.printf("[%lu] %-8s %.0f%% | raw:%.0f ADC:%.0f V:%.2f\n", 
      micros(), CLASS_NAMES[pred], conf*100, normalized*4095, adc_raw, voltage);
  }
}

void run_test_benchmarks() {
  Serial.println("--- Test Samples ---");
  int correct = 0;
  
  for (int s = 0; s < NUM_TEST_SAMPLES; s++) {
    fill_input(input_tensor, test_samples[s], N_INPUTS);
    interpreter->Invoke();
    
    float scores[N_OUTPUTS];
    read_output(output_tensor, scores, N_OUTPUTS);
    float probs[N_OUTPUTS];
    softmax(scores, probs, N_OUTPUTS);
    
    int pred = argmax(probs, N_OUTPUTS);
    int truth = test_labels[s];
    
    if (pred == truth) correct++;
    
    Serial.printf("Test %d/%d | True:%s | Pred:%s (%.0f%%)\n", 
      s+1, NUM_TEST_SAMPLES, CLASS_NAMES[truth], CLASS_NAMES[pred], probs[pred]*100);
  }
  
  float acc = correct * 100.0 / NUM_TEST_SAMPLES;
  Serial.printf("\nTest Accuracy: %.1f%% (%d/%d correct)\n", acc, correct, NUM_TEST_SAMPLES);
}

// Robust input/output handling
void fill_input(TfLiteTensor* tensor, const float* data, int len) {
  if (tensor->type == kTfLiteFloat32) {
    memcpy(tensor->data.f, data, len * sizeof(float));
  } else if (tensor->type == kTfLiteInt8) {
    float scale = tensor->params.scale;
    int zero_point = tensor->params.zero_point;
    for (int i = 0; i < len; i++) {
      int q = roundf(data[i] / scale) + zero_point;
      tensor->data.int8[i] = constrain(q, -128, 127);
    }
  }
}

void read_output(TfLiteTensor* tensor, float* out, int len) {
  if (tensor->type == kTfLiteFloat32) {
    memcpy(out, tensor->data.f, len * sizeof(float));
  } else if (tensor->type == kTfLiteInt8) {
    float scale = tensor->params.scale;
    int zero_point = tensor->params.zero_point;
    for (int i = 0; i < len; i++) {
      out[i] = (tensor->data.int8[i] - zero_point) * scale;
    }
  }
}

void softmax(float* in, float* out, int len) {
  float max = in[0];
  for (int i = 1; i < len; i++) if (in[i] > max) max = in[i];
  
  float sum = 0;
  for (int i = 0; i < len; i++) {
    out[i] = expf(in[i] - max);
    sum += out[i];
  }
  for (int i = 0; i < len; i++) out[i] /= sum;
}

int argmax(float* arr, int len) {
  int max_idx = 0;
  for (int i = 1; i < len; i++) {
    if (arr[i] > arr[max_idx]) max_idx = i;
  }
  return max_idx;
}

