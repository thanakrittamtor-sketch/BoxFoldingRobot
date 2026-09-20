from ultralytics import YOLO
import cv2

# โหลดโมเดลที่ฝึกเอง
model = YOLO(r"runs/detect/train-4/weights/best.pt")

# เปิดกล้อง
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # ตรวจจับ
    results = model(frame)

    # วาดกรอบ
    annotated = results[0].plot()

    cv2.imshow("Box Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()