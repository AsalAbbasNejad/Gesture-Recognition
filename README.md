# Gesture Recognition using TinyML on Arduino Nano 33 BLE Sense

This repository contains an end-to-end TinyML project focused on real-time gesture classification using motion sensors. The project demonstrates the full workflow from raw data collection to on-device inference.

## 🚀 Overview
The goal of this project is to recognize specific hand gestures (**Punch** and **Flex**) directly on the **Arduino Nano 33 BLE Sense** without relying on an external computer or cloud processing.

## 🛠 Hardware & Tools
* [cite_start]**Board**: Arduino Nano 33 BLE Sense Rev2 [cite: 1431]
* [cite_start]**Sensors**: BMI270 (Accelerometer & Gyroscope) [cite: 1434]
* [cite_start]**Frameworks**: TensorFlow, Keras, and TensorFlow Lite Micro (LiteRT) [cite: 1669, 1671]
* [cite_start]**Platform**: Google Colab for model training [cite: 1712]

## 📈 Project Workflow

### 1. Data Collection
We capture real-time sensor data from the accelerometer and gyroscope. 
* [cite_start]Data is recorded at a fixed rate (104 Hz)[cite: 1581, 1594].
* [cite_start]Gestures are stored as CSV files with labels: `punch.csv` and `flex.csv`[cite: 1703, 1707].

### 2. Preprocessing & Feature Extraction
To ensure high accuracy on low-power hardware, we process the data before feeding it to the model:
* [cite_start]**Normalization**: Scaling raw sensor values to a 0.0 - 1.0 range[cite: 1811].
* [cite_start]**Statistical Features**: Calculating **RMS** (Root Mean Square) for signal strength[cite: 3582].
* [cite_start]**Spectral Analysis**: Using **FFT** (Fast Fourier Transform) to analyze the frequency of movements[cite: 3666].

### 3. Model Training
A Deep Neural Network (DNN) was designed and trained in Python using Keras:
* [cite_start]**Input Layer**: Processes 714 normalized data points[cite: 2053].
* [cite_start]**Architecture**: Two hidden Dense layers (50 and 15 neurons)[cite: 2062, 2063].
* [cite_start]**Output Layer**: Softmax activation to predict the probability of each gesture[cite: 2064].

### 4. Deployment (TinyML)
The trained model is optimized for the edge:
* [cite_start]**Conversion**: The TensorFlow model is converted to a `.tflite` FlatBuffer format[cite: 2134, 2297].
* [cite_start]**C Array Generation**: Using `xxd` to convert the model into a C constant byte array (`model.h`)[cite: 2154, 2203].
* [cite_start]**Inference**: Running the model on the Arduino using the **LiteRT** library[cite: 2327, 2334].

## 📁 Structure
* `/data`: Collected CSV sensor datasets.
* `/notebook`: Training and conversion script (Jupyter Notebook).
* `/arduino_code`: Final C++ sketch for on-device detection.

## 🔧 Setup
1. Upload the `Nano33_IMU` sketch to capture your own data.
2. Train the model using the provided Google Colab notebook.
3. Replace the `model.h` file in the Arduino project with your generated model.
4. Upload and open the Serial Monitor to see real-time results.

---
**Course Project**: Low-power Embedded Systems  
[cite_start]**Instructor**: Prof. Kasım Sinan Yıldırım [cite: 4]
