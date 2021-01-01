import cv2
import os
import numpy as np

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/trainningData.yml")
id = 0
names = ["None", "vivek", "smit"]
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, img = cap.read(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y : y + h, x : x + w])
        if confidence < 100:
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x, y + 10), font, 0.55, (0, 255, 0), 1)
        # cv2.putText(img,str(confidence),(x+5,y+h-5),font,1,(255,255,0),1)
        cv2.imshow("Face", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        break
