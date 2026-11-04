// Trova LED ring firmware (optional): only if the ring around the arm camera is wired to an Arduino / ESP32
// you are allowed to flash. Library: Adafruit NeoPixel. Set DATA_PIN and N_PIXELS to the real wiring.
// Serial 115200, one command per line:
//   C r g b        fill the ring with this colour
//   F r g b n      flash n times, then back to the last fill colour
#include <Adafruit_NeoPixel.h>
#define DATA_PIN 6
#define N_PIXELS 16
Adafruit_NeoPixel ring(N_PIXELS, DATA_PIN, NEO_GRB + NEO_KHZ800);
uint32_t fillColor = 0;

void fillAll(uint32_t c) { for (int i = 0; i < N_PIXELS; i++) ring.setPixelColor(i, c); ring.show(); }

void setup() {
  Serial.begin(115200);
  ring.begin();
  ring.setBrightness(80);          // keep it modest: glare on shiny objects confuses the camera
  fillColor = ring.Color(90, 90, 90);
  fillAll(fillColor);
}

void loop() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  int r = 0, g = 0, b = 0, n = 0;
  if (line.startsWith("C") && sscanf(line.c_str(), "C %d %d %d", &r, &g, &b) == 3) {
    fillColor = ring.Color(r, g, b);
    fillAll(fillColor);
  } else if (line.startsWith("F") && sscanf(line.c_str(), "F %d %d %d %d", &r, &g, &b, &n) == 4) {
    for (int k = 0; k < n; k++) { fillAll(ring.Color(r, g, b)); delay(180); fillAll(0); delay(120); }
    fillAll(fillColor);
  }
}
