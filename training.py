from ultralytics import YOLO

def main():
    # Load the pretrained model
    model = YOLO("yolo11n.pt")

    # Train the model
    train_results = model.train(
        data="drowsiness.yaml",  # path to your dataset config
        epochs=100,
        imgsz=640,
        device=0,                # use GPU 0
        workers=0,               # VERY IMPORTANT on Windows
    )

    # Evaluate after training
    metrics = model.val()
    print("Validation results:", metrics)

if __name__ == "__main__":
    main()
