#include <Arduino.h>

// 定義電位器腳位（假設使用 Arduino Nano 的配置）
#define PIN_PINKY     A0
#define PIN_RING      A1
#define PIN_MIDDLE    A2
#define PIN_INDEX     A3
#define PIN_THUM      A4
#define PIN_JOY_X     A6
#define PIN_JOY_Y     A7

void setup() {
    Serial.begin(115200); // 初始化序列埠，波特率為 115200
    while (!Serial) {
        ; // 等待序列埠連接（某些板子需要）
    }
}

void loop() {
    // 讀取所有電位器的值
    int pinky_value = analogRead(PIN_PINKY);
    int ring_value = analogRead(PIN_RING);
    int middle_value = analogRead(PIN_MIDDLE);
    int index_value = analogRead(PIN_INDEX);
    int thumb_value = analogRead(PIN_THUM);
    int joy_x_value = analogRead(PIN_JOY_X);
    int joy_y_value = analogRead(PIN_JOY_Y);
    
    // 獲取當前時間（以毫秒為單位）
    unsigned long currentTime = millis();

    // 打印時間戳和電位器值，每行一個
    Serial.print("Time: ");
    Serial.print(currentTime);
    Serial.println(" ms");

    Serial.print("Pinky: ");
    Serial.println(pinky_value);

    Serial.print("Ring: ");
    Serial.println(ring_value);

    Serial.print("Middle: ");
    Serial.println(middle_value);

    Serial.print("Index: ");
    Serial.println(index_value);

    Serial.print("Thumb: ");
    Serial.println(thumb_value);

    Serial.print("Joy X: ");
    Serial.println(joy_x_value);

    Serial.print("Joy Y: ");
    Serial.println(joy_y_value);

    Serial.println("----------------"); // 分隔線，方便閱讀

    delay(1000); // 每秒打印一次（可調整延遲時間）
}
