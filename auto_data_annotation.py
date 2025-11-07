import os
from ultralytics import YOLO
import cv2

# Paths
images_path = "data/images"
labels_path = "data/labels"
previews_path = "data/previews"

os.makedirs(labels_path, exist_ok=True)
os.makedirs(previews_path, exist_ok=True)

model = YOLO("yolov11n-face.pt")

for img_file in os.listdir(images_path):
    if not img_file.endswith(".jpg"):
        continue

    img_path = os.path.join(images_path, img_file)
    img = cv2.imread(img_path)
    h_img, w_img = img.shape[:2]

    # Run detection
    results = model(img)

    label_lines = []

    for r in results:
        for box in r.boxes.xyxy.cpu().numpy():  # xyxy format: x1, y1, x2, y2
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            x_center = (x1 + w / 2) / w_img
            y_center = (y1 + h / 2) / h_img
            width = w / w_img
            height = h / h_img
            class_id = 0 if "awake" in img_file else 1
            label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Draw bounding box on preview
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, "awake" if class_id == 0 else "drowsy", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Save label file
    label_file = os.path.join(labels_path, img_file.replace(".jpg", ".txt"))
    with open(label_file, "w") as f:
        f.write("\n".join(label_lines))

    # Save preview
    preview_file = os.path.join(previews_path, img_file)
    cv2.imwrite(preview_file, img)

print("Done! Labels and previews saved.")
