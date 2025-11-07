import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
model = YOLO("yolo11n.pt")

if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

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