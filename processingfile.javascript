import processing.serial.*;

Serial myPort;
float pinkyAngle, ringAngle, middleAngle, indexAngle, thumbAngle;

void setup() {
  size(800, 600);
  background(255);

  // 列出所有可用序列埠
  printArray(Serial.list());
  
  // 連接到 COM9
  String portName = "COM9";
  try {
    myPort = new Serial(this, portName, 115200);
    myPort.bufferUntil('\n');
    println("成功連接到 " + portName);
  } catch (Exception e) {
    println("無法連接到 " + portName + ": " + e.getMessage());
    exit();
  }
}

void draw() {
  background(255); // 白色背景

  // 繪製手掌（帶圓角的矩形）
  fill(220, 180, 140); // 自然膚色
  noStroke();
  rect(300, 400, 200, 100, 20); // 圓角矩形手掌

  // 繪製手指（從手掌頂部開始）
  drawFinger(320, 400, pinkyAngle, false);  // 小指
  drawFinger(360, 400, ringAngle, false);   // 無名指
  drawFinger(400, 400, middleAngle, false); // 中指
  drawFinger(440, 400, indexAngle, false);  // 食指
  drawFinger(480, 400, thumbAngle, true);   // 拇指（特別處理）
}

void serialEvent(Serial myPort) {
  String data = myPort.readStringUntil('\n');
  if (data != null) {
    data = trim(data);
    String[] values = split(data, ',');
    if (values.length == 5) {
      pinkyAngle = map(float(values[0]), 0, 1023, 0, -PI/2);
      ringAngle = map(float(values[1]), 0, 1023, 0, -PI/2);
      middleAngle = map(float(values[2]), 0, 1023, 0, -PI/2);
      indexAngle = map(float(values[3]), 0, 1023, 0, -PI/2);
      thumbAngle = map(float(values[4]), 0, 1023, 0, -PI/2);
    }
  }
}

void drawFinger(float x, float y, float angle, boolean isThumb) {
  float fingerWidth = 20; // 手指寬度
  float segmentLength = 40; // 每節長度（拇指稍短）

  fill(220, 180, 140); // 膚色
  noStroke();

  pushMatrix();
  translate(x, y);

  // 第一節（第一掌骨）
  if (isThumb) {
    rotate(-PI/4); // 拇指從側面開始
    rect(0, -segmentLength, fingerWidth, segmentLength, 10); // 圓角矩形
    translate(0, -segmentLength);
    rotate(angle * 0.5); // 拇指第一關節彎曲
  } else {
    rect(0, -segmentLength, fingerWidth, segmentLength, 10);
    translate(0, -segmentLength);
    rotate(angle * 0.3); // 第一關節彎曲 30%
  }

  // 第二節（第二掌骨）
  rect(0, -segmentLength, fingerWidth, segmentLength, 10);
  translate(0, -segmentLength);
  rotate(angle * 0.5); // 第二關節彎曲 50%

  // 第三節（第三掌骨，較短）
  rect(0, -segmentLength * 0.75, fingerWidth, segmentLength * 0.75, 10);

  // 添加簡單指甲（第三節頂部）
  fill(240, 220, 200); // 指甲顏色
  rect(0, -segmentLength * 0.75, fingerWidth * 0.6, 10, 5);

  popMatrix();
}
