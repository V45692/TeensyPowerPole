#include <ADC.h>

// 14-byte packet: Header(4), S1(2), S2(2), S3(2), TS(4)
struct __attribute__((packed)) DataPacket {
  uint32_t sync = 0xFFFFFFFF; 
  uint16_t s1;
  uint16_t s2;
  uint16_t s3;
  uint32_t ts;
};

ADC *adc = new ADC();
DataPacket packet;

void setup() {
  pinMode(2, INPUT_PULLUP);
  Serial.begin(1); 

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
  if (digitalRead(2) == HIGH) {
    Serial.println("START"); // One-time sync string
    uint32_t recordStart = millis();

    while (millis() - recordStart < 5000) {
      // Direct fast reads
      packet.s1 = adc->adc0->analogRead(A9);
      packet.s2 = adc->adc0->analogRead(A8);
      packet.s3 = adc->adc1->analogRead(A7);
      packet.ts = micros();

      Serial.write((uint8_t*)&packet, sizeof(packet));
    }
    Serial.println("STOP");
    while(1); 
  }
}