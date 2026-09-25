import cv2
from ultralytics import YOLO

# Load the combined Key + Wallet model
model = YOLO("runs/detect/combined_yolo26n/weights/best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera active. Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO26 NMS-free inference
    results = model(frame, conf=0.5, nms=False)

    annotated_frame = results[0].plot()

    cv2.imshow(
        "YOLO26n Combined Object Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()