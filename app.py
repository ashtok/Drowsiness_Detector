import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load your trained YOLO model

print(gr.__version__)
print(dir(gr))
print(hasattr(gr, "Camera"))

model = YOLO("runs/detect/train2/weights/best.pt")

def detect(frame):
    """Runs YOLO on webcam frames and returns annotated image."""
    if frame is None:
        return None
    if isinstance(frame, Image.Image):  # if Gradio gives a PIL image
        frame = np.array(frame)
    results = model(frame)
    result_image = results[0].plot()
    return Image.fromarray(result_image)

# Build Gradio interface with live webcam
demo = gr.Interface(
    fn=detect,
    inputs = gr.Image(sources="webcam", streaming=True),  # ✅ supported in Gradio 5.49+
    outputs=gr.Image(label="Detection Result"),
    live=True,
    title="Drowsiness Detector",
    description="Real-time drowsiness detection using your webcam powered by YOLO.",
)

if __name__ == "__main__":
    demo.launch()
