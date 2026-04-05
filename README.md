# Gesture Recognition using TinyML on Arduino Nano 33 BLE Sense


This repository demonstrates a complete end-to-end TinyML pipeline for real-time gesture classification. Using an **Arduino Nano 33 BLE Sense**, the system identifies specific hand movements like **Punch** and **Flex** by processing motion data directly on the microcontroller.

## 🚀 Project Overview
The goal is to implement "Intelligence at the Edge." By running Machine Learning models on-device, we eliminate the need for cloud communication, ensuring:
* **Low Latency**: Immediate detection and response.
* **Privacy**: Sensor data is processed locally and never uploaded.
* **Efficiency**: Optimized for low-power embedded environments.

## 🛠 Hardware & Technologies
* **Board**: Arduino Nano 33 BLE Sense Rev2
* **MCU**: Arm Cortex-M4 (64 MHz) with 1MB Flash and 256KB RAM
* **IMU Sensor**: BMI270 (6-axis Accelerometer and Gyroscope)
* **Software**: TensorFlow, Keras, and TensorFlow Lite Micro (LiteRT)
* **Training Platform**: Google Colab (Jupyter Notebooks)

## 📈 Deployment Workflow

### 1. Data Collection
Motion data is captured at 104 Hz. Two gestures were recorded and stored in CSV format:
* **Punch**: A rapid forward strike.
* **Flex**: A bicep-curling motion.

### 2. Preprocessing & Feature Extraction
To maintain high performance on a low-power chip, the raw data is transformed into meaningful features:
* **Normalization**: Scaling accelerometer and gyroscope values to a 0.0 – 1.0 range.
* **RMS Calculation**: Extracting Root Mean Square values to represent signal magnitude.
* **FFT/PSD**: Performing Fast Fourier Transform to analyze the frequency components of each gesture.

### 3. Neural Network Architecture
The model is a Deep Neural Network (DNN) built with Keras:
* **Input**: 714 normalized data points (2-second window).
* **Layers**: Two Dense (Fully Connected) hidden layers with 50 and 15 neurons respectively.
* **Activation**: ReLU for hidden layers and Softmax for the final output.

### 4. Edge Implementation
* **Conversion**: The model is compressed into `.tflite` format.
* **Serialization**: The `.tflite` file is converted into a C constant byte array (`model.h`).
* **Inference**: The **LiteRT** library executes the model on the Arduino without dynamic memory allocation.

## 📁 Repository Structure
* `/data`: Contains CSV files of recorded gestures.
* `/notebooks`: Python training and conversion scripts.
* `/arduino_inference`: Final C++ code for real-time gesture recognition.

---
**Course**: Low-power Embedded Systems  
**Instructor**: Prof. Kasım Sinan Yıldırım
