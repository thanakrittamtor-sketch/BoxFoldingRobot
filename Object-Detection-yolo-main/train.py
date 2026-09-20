from ultralytics import YOLO

# โหลดโมเดลเริ่มต้น
model = YOLO("yolo11n.pt")

# เริ่มฝึกโมเดล
model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8
)