from ultralytics import YOLO
model = YOLO("best.pt")
result = model.predict(
    source="test/images",
    mode="predict",
    save=True,
    device="cpu"
)