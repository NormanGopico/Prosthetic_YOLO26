from ultralytics import YOLO

def main():
    model = YOLO("yolo26n.pt")

    model.train(
        data="dataset_wallet/data.yaml",
        epochs=15,
        imgsz=640,
        batch=16,
        name="wallet_yolo26n"
    )

if __name__ == "__main__":
    main()