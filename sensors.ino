#include <Servo.h>

const int Flex_PIN = 8;
const float VCC = 5000.0;
const float Voltage_0 = 2000.0; // 需校準
const float Voltage_set = 3000.0; // 需校準
const float Angle_0 = 0.0;
const float Angle_set = 90.0;

#define K_Value (Angle_set - Angle_0) / (Voltage_set / 1000 - Voltage_0 / 1000)
#define B_Value Angle_0 - (Voltage_0 / 1000) * K_Value

Servo servos[5];
int servoPins[] = {3, 5, 6, 9, 10};

const unsigned long collectionTime = 10000;
unsigned long startTime;
bool collecting = false;
String dataBuffer = "";

void setup() {
  Serial.begin(9600);
  pinMode(Flex_PIN, INPUT);
  for (int i = 0; i < 5; i++) {
    servos[i].attach(servoPins[i]);
  }
  Serial.println("Arduino Ready");
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    Serial.print("收到命令: "); // 調試：顯示接收到的命令
    Serial.println(command);
    if (command == "START") {
      startCollection();
    } else if (command.startsWith("ANGLES:")) {
      setServoAngles(command);
    }
  }

  if (collecting) {
    collectData();
  }
}

void startCollection() {
  collecting = true;
  startTime = millis();
  dataBuffer = "";
  Serial.println("Started collecting data");
}

void collectData() {
  if (millis() - startTime < collectionTime) {
    int F_ADC = analogRead(Flex_PIN);
    float Flex_V = F_ADC * VCC / 1024.0;
    float angle = K_Value * (Flex_V / 1000) + B_Value;
    if (angle > 180) angle = 180;
    if (angle < -180) angle = -180;
    dataBuffer += String(angle) + ",";
    delay(100);
  } else {
    collecting = false;
    Serial.println("DATA:" + dataBuffer);
    dataBuffer = "";
  }
}

void setServoAngles(String command) {
  command.remove(0, 7); // 移除 "ANGLES:"
  int angles[5];
  int index = 0;
  char *token = strtok(command.c_str(), ",");
  while (token != NULL && index < 5) {
    angles[index] = atoi(token);
    Serial.print("Angle ");
    Serial.print(index);
    Serial.print(": ");
    Serial.println(angles[index]);
    index++;
    token = strtok(NULL, ",");
  }
  for (int i = 0; i < 5; i++) {
    servos[i].write(angles[i]);
  }
  Serial.println("Servos set to new angles");
}