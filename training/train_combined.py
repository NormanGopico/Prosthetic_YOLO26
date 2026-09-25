from ultralytics import YOLO

def main():
    model = YOLO("yolo26n.pt")

    model.train(
        data="dataset_combined/data.yaml",
        epochs=15,
        imgsz=640,
        batch=16,
        name="combined_yolo26n",
        exist_ok=True
    )

if __name__ == "__main__":
    main()