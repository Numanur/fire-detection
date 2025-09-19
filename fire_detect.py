import cv2
from ultralytics import YOLO
import time

last_alarm_time = 0.0
alarm_cooldown = 3.0 

def play_alarm():
    """Play a short alarm sound, cross-platform, with cooldown."""
    global last_alarm_time
    now = time.time()
    if now - last_alarm_time < alarm_cooldown:
        return
    last_alarm_time = now

    try:
        import winsound
        for _ in range(2):
            winsound.Beep(1200, 200) 
            winsound.Beep(800, 200)
        return
    except Exception:
        pass

    try:
        import numpy as np
        import simpleaudio as sa
        sr = 44100
        dur = 0.6 
        f1, f2 = 1200, 800

        t = np.linspace(0, dur/2, int(sr * (dur/2)), False)
        tone1 = 0.5 * np.sin(2 * np.pi * f1 * t)
        tone2 = 0.5 * np.sin(2 * np.pi * f2 * t)

        audio = np.concatenate([tone1, tone2]).astype(np.float32)
        audio_i16 = (audio * 32767).astype(np.int16)
        sa.play_buffer(audio_i16, 1, 2, sr)
        return
    except Exception:
        pass

    try:
        print('\a', end='', flush=True)
    except Exception:
        pass

model = YOLO('best_2.pt') 

cap = cv2.VideoCapture("http://10.50.117.225:8080/video")

prev_time = 0.0
delay = 0.5 
class_names = {0: "fire", 1: "smoke"}

ALERT_CONF_THRESH = 0.25

annotated_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    if annotated_frame is None:
        annotated_frame = frame.copy()

    current_time = time.time()
    if current_time - prev_time >= delay:
        results = model(frame)
        annotated_frame = results[0].plot()

        fire_area_total = 0
        smoke_area_total = 0
        any_alert = False

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            cls_arr = boxes.cls.detach().cpu().numpy().astype(int)
            conf_arr = boxes.conf.detach().cpu().numpy()
            xyxy_arr = boxes.xyxy.detach().cpu().numpy().astype(int)

            for (cls_id, conf, (x1, y1, x2, y2)) in zip(cls_arr, conf_arr, xyxy_arr):
                area = (x2 - x1) * (y2 - y1)

                if cls_id == 0:
                    fire_area_total += area
                    color = (0, 0, 255)        
                else:
                    smoke_area_total += area
                    color = (0, 165, 255)  

                if conf >= ALERT_CONF_THRESH:
                    any_alert = True

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated_frame,
                    f"{class_names.get(cls_id, 'obj')} {conf:.2f}: {int(area)} px^2",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        print(f"Total Fire Area (px^2): {fire_area_total}")
        print(f"Total Smoke Area (px^2): {smoke_area_total}")

        if any_alert:
            play_alarm()

        prev_time = current_time

    cv2.imshow("Fire and Smoke Detection - IP Camera", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
