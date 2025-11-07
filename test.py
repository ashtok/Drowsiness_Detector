from ultralytics import YOLO
import cv2 as cv2
import numpy as np

model = YOLO("runs/detect/train2/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()#
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Error: Failed to read frame.")
        break
    result = model(frame)

    cv2.imshow("Yolo 11 Detector", result[0].plot())

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()