#include "Arduino_BMI270_BMM150.h"
#include "model.h"

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <math.h>

constexpr int NUM_SAMPLES = 128;
constexpr int NUM_AXES = 6;
constexpr int NUM_FEATURES = 42;
constexpr int NUM_CLASSES = 4;

const float ACC_THRESHOLD = 1.0f;

const char* CLASS_NAMES[NUM_CLASSES] = {
  "circle",
  "left_right",
  "rest",
  "up_down"
};

const float SCALER_MEAN[NUM_FEATURES] = {
 0.0028456297841330525, 0.15323115103607657, 0.20991249466411924, -0.21135899900400545, 0.24983598484444278, 4.793294270833333, 0.010060067997470981, -0.09267954183451366, 0.16192201818921603, 0.4710910622185717, -0.4333190759255861, 0.24294787406688556, 4.264322916666667, 0.008400465739974744, -0.13490075709220642, 0.07853449013509817, 0.6600663011583189, -0.32984546727190417, 0.06030338545194051, 14.794921875, 0.0012193116736847938, -1.635650815597425, 15.408058225216033, 15.744528147391975, -36.05111258399362, 31.64354811546703, 7.218424479166667, 60.0580913823408, 0.20706652270261353, 12.766421422866793, 12.910864355391823, -29.940286095409345, 25.015511575543012, 9.814453125, 44.01413348610807, 0.6706116674467921, 41.08685917087132, 41.79945999151096, -65.77205386742328, 82.43560286436696, 5.029296875, 708.5841218412396
};

const float SCALER_SCALE[NUM_FEATURES] = {
0.14288585025748984, 0.15666630159036002, 0.1561580142293484, 0.1422546317480074, 0.34996500407577597, 5.846469937476034, 0.015037652823402633, 0.5662861012228516, 0.11870323264624183, 0.3842549288864529, 0.6743758436412166, 0.48786658744500633, 4.63006900539109, 0.009627278821093948, 0.7439934317622383, 0.05842360383363899, 0.38159913452583893, 0.8301282711825985, 0.6660625563706777, 9.224467255768793, 0.0013611520532262557, 3.582872136335081, 10.645377517472731, 10.879084622439816, 28.15112142542858, 21.526717048353674, 10.205736803196519, 60.2045336113747, 2.383320631369149, 12.427382935021077, 12.50815898693468, 25.69225849353342, 20.277745969413413, 10.099005150977467, 71.98670820713639, 10.203213655247504, 37.16534456856686, 37.772387600173744, 55.68881927706673, 79.51543212737768, 7.128961989650812, 996.7471063415105
};

float sampleBuffer[NUM_SAMPLES][NUM_AXES];

tflite::AllOpsResolver resolver;
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* model_tflite = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int tensorArenaSize = 16 * 1024;
uint8_t tensorArena[tensorArenaSize];

float computeMean(const float* x, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; i++) s += x[i];
  return s / n;
}

float computeStd(const float* x, int n, float meanVal) {
  float s = 0.0f;
  for (int i = 0; i < n; i++) {
    float d = x[i] - meanVal;
    s += d * d;
  }
  return sqrtf(s / n);
}

float computeRMS(const float* x, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; i++) s += x[i] * x[i];
  return sqrtf(s / n);
}

float computeMin(const float* x, int n) {
  float m = x[0];
  for (int i = 1; i < n; i++) {
    if (x[i] < m) m = x[i];
  }
  return m;
}

float computeMax(const float* x, int n) {
  float m = x[0];
  for (int i = 1; i < n; i++) {
    if (x[i] > m) m = x[i];
  }
  return m;
}

void computeFrequencyFeatures(const float* x, int n, float fs, float& dominantFreq, float& peakPower) {
  const int maxBin = 16;
  peakPower = 0.0f;
  dominantFreq = 0.0f;

  for (int k = 1; k <= maxBin; k++) {
    float realPart = 0.0f;
    float imagPart = 0.0f;

    for (int t = 0; t < n; t++) {
      float angle = 2.0f * PI * k * t / n;
      realPart += x[t] * cosf(angle);
      imagPart -= x[t] * sinf(angle);
    }

    float power = realPart * realPart + imagPart * imagPart;
    if (power > peakPower) {
      peakPower = power;
      dominantFreq = (fs * k) / n;
    }
  }
}

void extractFeatures(float features[NUM_FEATURES]) {
  float axisSignal[NUM_SAMPLES];
  int idx = 0;

  for (int axis = 0; axis < NUM_AXES; axis++) {
    for (int i = 0; i < NUM_SAMPLES; i++) {
      axisSignal[i] = sampleBuffer[i][axis];
    }

    float meanVal = computeMean(axisSignal, NUM_SAMPLES);
    float stdVal = computeStd(axisSignal, NUM_SAMPLES, meanVal);
    float rmsVal = computeRMS(axisSignal, NUM_SAMPLES);
    float minVal = computeMin(axisSignal, NUM_SAMPLES);
    float maxVal = computeMax(axisSignal, NUM_SAMPLES);

    float dominantFreq, peakPower;
    computeFrequencyFeatures(axisSignal, NUM_SAMPLES, 100.0f, dominantFreq, peakPower);

    features[idx++] = meanVal;
    features[idx++] = stdVal;
    features[idx++] = rmsVal;
    features[idx++] = minVal;
    features[idx++] = maxVal;
    features[idx++] = dominantFreq;
    features[idx++] = peakPower;
  }
}

void normalizeFeatures(float features[NUM_FEATURES]) {
  for (int i = 0; i < NUM_FEATURES; i++) {
    if (SCALER_SCALE[i] != 0.0f) {
      features[i] = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }
  }
}

int argmaxOutput() {
  int best = 0;
  for (int i = 1; i < NUM_CLASSES; i++) {
    if (output->data.f[i] > output->data.f[best]) {
      best = i;
    }
  }
  return best;
}

bool waitForMotionTrigger() {
  float aX, aY, aZ;

  while (true) {
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

      if (aSum >= ACC_THRESHOLD) {
        delay(50);
        return true;
      }
    }
  }
}

bool captureWindow() {
  float aX, aY, aZ, gX, gY, gZ;
  int samplesRead = 0;

  while (samplesRead < NUM_SAMPLES) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      sampleBuffer[samplesRead][0] = aX;
      sampleBuffer[samplesRead][1] = aY;
      sampleBuffer[samplesRead][2] = aZ;
      sampleBuffer[samplesRead][3] = gX;
      sampleBuffer[samplesRead][4] = gY;
      sampleBuffer[samplesRead][5] = gZ;

      samplesRead++;
    }
  }

  return true;
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("Starting final gesture classifier...");

  if (!IMU.begin()) {
    Serial.println("IMU failed");
    while (1);
  }

  model_tflite = tflite::GetModel(gesture_model_tflite);

  static tflite::MicroInterpreter static_interpreter(
      model_tflite,
      resolver,
      tensorArena,
      tensorArenaSize,
      error_reporter,
      nullptr);

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model ready!");
}

void loop() {
  if (!waitForMotionTrigger()) return;
  if (!captureWindow()) return;

  float features[NUM_FEATURES];
  extractFeatures(features);
  normalizeFeatures(features);

  for (int i = 0; i < NUM_FEATURES; i++) {
    input->data.f[i] = features[i];
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    delay(1000);
    return;
  }

  int pred = argmaxOutput();

  Serial.print("Prediction: ");
  Serial.println(CLASS_NAMES[pred]);

  Serial.print("circle: ");
  Serial.print(output->data.f[0], 6);
  Serial.print(" left_right: ");
  Serial.print(output->data.f[1], 6);
  Serial.print(" rest: ");
  Serial.print(output->data.f[2], 6);
  Serial.print(" up_down: ");
  Serial.println(output->data.f[3], 6);

  Serial.println();
  delay(1000);
}