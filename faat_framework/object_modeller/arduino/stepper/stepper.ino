


void setup() {
  Serial.begin(9600);
  pinMode(12, OUTPUT);
  pinMode(9, OUTPUT);
  
  pinMode(4, INPUT);
  pinMode(7, INPUT);
}

const int nrTotalSteps = 98;
unsigned int stepSize = nrTotalSteps / 10;

void step()
{
    int last_a = 0;
    int last_b = 0;
    unsigned int step = 0;
    
    // start the motor
    digitalWrite(12, HIGH); // forward
    digitalWrite(9, LOW); // disengage brake;
    analogWrite(3, 39); // speed
    
    while (step < stepSize)
    {
      int a = digitalRead(4);
      int b = digitalRead(7);
    
      if (last_a != a || last_b != b)
      {
        last_a = a;
        last_b = b;
        step++;
      }
    }
    
    // brake
    digitalWrite(9, HIGH); // brake
    analogWrite(3, 0);
    
}

void loop() {
  
  while (Serial.available())
  {
    int command = Serial.read();
    
    if (command == 'w')
    {
      int stepWidthDeg = Serial.parseInt();
      unsigned long test = nrTotalSteps * stepWidthDeg;
      stepSize = test / 36;
      Serial.print('k');
    }
    
    if (command == 'a')
    {
      Serial.print('k');
    }
    
    if (command == 's')
    {
      step();
      delay(2000);
      Serial.print('k');
    }
  }
  
    /*
  while (Serial.available())
  {
    int a = digitalRead(4);
    int b = digitalRead(7);
    
    if (last_a != a || last_b != b)
    {
      last_a = a;
      last_b = b;
      step++;
      if (step > 979)
      {
        step = 0;
      }
      Serial.println(step);
    }
    */
    
    //Serial.print("Val: ");
    //Serial.print(a);
    //Serial.print(" ");
    //Serial.println(b);
    /*
    digitalWrite(12, HIGH); // forward
    digitalWrite(9, LOW); // disengage brake;
    analogWrite(3, 50); // speed
    
    delay(1000);
    
    digitalWrite(9, HIGH); // brake
    analogWrite(3, 1);
    
    delay(1000);
    */
  //}
    
    //digitalWrite(9, HIGH); // brake
    //analogWrite(3, 1);
}
