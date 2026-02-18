#include <ADC.h>

// 8-byte packet (Aligned for 32-bit speed)
struct __attribute__((packed)) DataPacket {
  uint16_t s1; // Impact (0°) - ADC0
  uint16_t s2; // Side (90°)   - ADC0
  uint16_t s3; // Opp (180°)    - ADC1
  uint16_t padding; 
};

ADC *adc = new ADC();
IntervalTimer timer;
DataPacket packet;
volatile bool capturing = false;

// This function runs every 3.33 microseconds
void captureISR() {
  // Use 'auto' so the compiler handles the struct naming for you
  auto res = adc->analogSynchronizedRead(A9, A2);
  
  // Extract S1 (ADC0) and S3 (ADC1)
  packet.s1 = (uint16_t)res.result_adc0;
  packet.s3 = (uint16_t)res.result_adc1;
  
  // Read S2 (Side sensor) on ADC0
  packet.s2 = (uint16_t)adc->adc0->analogRead(A8);
  
  // Send the 8-byte packet to Python
  Serial.write((uint8_t*)&packet, sizeof(packet));
}

void setup() {
  Serial.begin(1); // Teensy goes as fast as USB allows
  pinMode(2, INPUT_PULLUP);

  // Overclock ADC settings for 3.3us cycle
  adc->adc0->setAveraging(0);
  adc->adc0->setResolution(10);
  adc->adc0->setConversionSpeed(ADC_CONVERSION_SPEED::VERY_HIGH_SPEED);
  adc->adc0->setSamplingSpeed(ADC_SAMPLING_SPEED::VERY_HIGH_SPEED);

  adc->adc1->setAveraging(0);
  adc->adc1->setResolution(10);
  adc->adc1->setConversionSpeed(ADC_CONVERSION_SPEED::VERY_HIGH_SPEED);
  adc->adc1->setSamplingSpeed(ADC_SAMPLING_SPEED::VERY_HIGH_SPEED);
}

void loop() {
  // Trigger on Pin 2
  if (digitalRead(2) == LOW && !capturing) {
    Serial.println("START");
    capturing = true;
    timer.begin(captureISR, 3.33); // 300 kHz
    
    delay(2000); // Record for 2 seconds
    
    timer.end();
    capturing = false;
    Serial.println("STOP");
  }
}