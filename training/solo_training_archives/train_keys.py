from ultralytics import YOLO

def main():
    # Load the official pretrained YOLO26n detection model
    model = YOLO("yolo26n.pt")

    # Fine-tune YOLO26n on the keys dataset
    model.train(
        data="dataset_keys/data.yaml",
        epochs=15,
        imgsz=640,
        batch=16,
        name="keys_yolo26n"
    )

if __name__ == "__main__":
    main()