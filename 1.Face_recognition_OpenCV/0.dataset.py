import cv2
import os
import numpy as np

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

id = input("enter user id")
sampleNumber = 0
while True:
    ret, img = cap.read(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        sampleNumber = sampleNumber + 1
        cv2.imwrite(
            "dataSet/User." + str(id) + "." + str(sampleNumber) + ".jpg",
            gray[y : y + h, x : x + w],
        )
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.waitKey(100)
        cv2.imshow("Face", img)
        cv2.waitKey(1)
        if sampleNumber > 15:
            cap.release()
            cv2.destroyAllWindows()
        break
