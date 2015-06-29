
char incomingByte; 


void setup()
{
  Serial.begin(115200);
  Serial.println("Testing Smart");
  pinMode(10,OUTPUT);
  pinMode(11,OUTPUT);
  pinMode(5,OUTPUT);
  pinMode(6,OUTPUT);
  pinMode(13,OUTPUT);
 
}

void loop()
{
   if (Serial.available() >0) {
    // read the oldest byte in the serial buffer:
   char incomingByte = Serial.read();
    
    if(incomingByte=='1') // Gesture A -> move forward
    {
       // digitalWrite(13, HIGH-digitalRead(13));
       analogWrite(5,100);
       analogWrite(6,100);
       digitalWrite(10, HIGH);
       digitalWrite(11, HIGH);
       //delay(3000);
     //  analogWrite(5,0);
    //   analogWrite(6,0);
       
    }
    else if(incomingByte=='2') // Gesture B -> move Backward
    {
      analogWrite(5,100);
       analogWrite(6,100);
       digitalWrite(10, LOW);
       digitalWrite(11, LOW);
       //delay(3000);
       //analogWrite(5,0);
       //analogWrite(6,0);
    }
    else if(incomingByte=='3') // Gesture C - > move clockwise
    {
      analogWrite(5,100);
       analogWrite(6,100);
       digitalWrite(10, LOW);
       digitalWrite(11, HIGH);
       delay(2000);
       analogWrite(5,0);
       analogWrite(6,0);
         
    }
    
     else if(incomingByte=='4') // Gesture FIVE - > Move anticlockwise
    {
     analogWrite(5,100);
       analogWrite(6,100);
       digitalWrite(10, HIGH);
       digitalWrite(11, LOW);
       delay(2000);
       analogWrite(5,0);
       analogWrite(6,0);
    
       
    }
    
    else if(incomingByte=='5') // // Gesture Point  - > STOP
    {
     //  analogWrite(5,100);
       //analogWrite(6,100);
       //digitalWrite(10, HIGH);
       //digitalWrite(11, LOW);
       //delay(6000);
        analogWrite(5,0);
       analogWrite(6,0);     
    }
    
    else if(incomingByte=='6') // // Gesture V  - > Check arduino
    {
       
       
       digitalWrite(13 , HIGH - digitalRead(13));
    }
    
    
    
   // Serial.println("Gesture \n"+incomingByte);
    
    //if(incomingByte=='1')
    //{
      //Serial.println("Gesture "+incomingByte);
      //digitalWrite(13, HIGH-digitalRead(13));
      //delay(1000);
    //}
    
    
}
}
