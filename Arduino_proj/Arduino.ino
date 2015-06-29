
void setup()
{
  Serial.begin(115200);
  pinMode(13,OUTPUT);
}

void loop()
{
  if(Serial.available())
  {
    char ch = Serial.read();
    
    switch(ch)
    {
      case '1':
      digitalWrite(13,HIGH-digitalRead(13));
      break;
    }
  }
}
  
  
  



