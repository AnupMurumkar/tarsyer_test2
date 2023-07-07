import cv2
import time
import numpy as np

car_classifier = cv2.CascadeClassifier(r'haarcascade_car.xml')  #path of haarcascade file

cap = cv2.VideoCapture(r'cars_on_highway (360p).mp4')  #path of video file

count =0
while cap.isOpened():
    
    time.sleep(.05)

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    cars = car_classifier.detectMultiScale(gray, 1.4, 2)
    
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        count +=1
        cv2.imshow('Cars', frame)
        print(count,'cars found')
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()
