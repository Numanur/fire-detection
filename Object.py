import cv2
from ultralytics import YOLO
import time

model = YOLO('best.pt')  # your trained model

cap = cv2.VideoCapture("http://192.168.1.107:8080/video")

prev_time = 0
delay = 1  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    current_time = time.time()
    if current_time - prev_time >= delay:
        results = model(frame)
        annotated_frame = results[0].plot()
        prev_time = current_time
    else:
        # Skip detection, just show previous annotated frame
        pass

    cv2.imshow("Fire Detection - IP Camera", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
