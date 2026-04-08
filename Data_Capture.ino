#include "Arduino_BMI270_BMM150.h"

const float accelerationThreshold = 1.3;
const int numSamples = 128;
const bool collectRestData = true;

int samplesRead = numSamples;

void setup() {
  Serial.begin(9600);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  Serial.println("aX,aY,aZ,gX,gY,gZ");
}

void loop() {
  float aX, aY, aZ, gX, gY, gZ;

  while (!collectRestData && samplesRead == numSamples) {
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);

      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

      if (aSum >= accelerationThreshold) {
        samplesRead = 0;
        delay(100);
        break;
      }
    }
  }

  if (collectRestData && samplesRead == numSamples) {
    samplesRead = 0;
    delay(500);
  }

  while (samplesRead < numSamples) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      Serial.print(aX, 6); Serial.print(',');
      Serial.print(aY, 6); Serial.print(',');
      Serial.print(aZ, 6); Serial.print(',');
      Serial.print(gX, 6); Serial.print(',');
      Serial.print(gY, 6); Serial.print(',');
      Serial.println(gZ, 6);

      samplesRead++;

      if (samplesRead == numSamples) {
        Serial.println();
        delay(700);
      }
    }
  }
}