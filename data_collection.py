import uuid
import os
import time
import cv2 as cv

data_folder = os.path.dirname(os.path.abspath(__file__))
print(data_folder)
images_folder = "data/images"
new_data_path = os.path.join(data_folder, images_folder)
labels = ["awake", "drowsy"]
custom_images = 150
print(new_data_path)

cap = cv.VideoCapture(0)
for label in labels:
    print(f"Collecting images for {label} label")
    time.sleep(5)
    for img_num in range(custom_images):
        print(f"Collecting images for {label} label, image num {img_num}")
        ret, frame = cap.read()
        frame = cv.resize(frame, (640, 480))
        frame =cv.flip(frame,1)

        img_name = os.path.join(new_data_path, label + '.' + str(uuid.uuid1()) + '.jpg')
        print(f"Writing image to {img_name}")

        cv.putText(frame, f"{label} {img_num + 1}/{custom_images}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imwrite(img_name, frame)
        cv.imshow("Image Collection", frame)

        if cv.waitKey(100) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
