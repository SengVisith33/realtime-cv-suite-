import cv2
import subprocess
import time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

target_object = None
last_spoken_time = 0

def speak_mac(text):
    subprocess.Popen(["say", text])

print("READY! Click the video window, press SPACE to type an item, or Q to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w, _ = frame.shape
    status_text = f"target: {target_object or 'none (press SPACE)'}"

    if target_object:
        results = model(frame, verbose=False)
        for box in results[0].boxes:
            label = model.names[int(box.cls[0])]
            if label == target_object:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_center_x = (x1 + x2) / 2
                offset = (box_center_x - (frame_w / 2)) / (frame_w / 2)

                direction = "left" if offset < -0.2 else ("right" if offset > 0.2 else "centered")
                status_text = f"{label}: {direction.upper()} off={offset:+.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                if time.time() - last_spoken_time > 3.0:
                    speak_mac(f"{label} is {direction}")
                    last_spoken_time = time.time()
                break

    cv2.putText(frame, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2)
    cv2.imshow("Mac Object Finder", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        speak_mac("Type what you want to find")
        user_input = input("\nType an object to find (e.g. cell phone, bottle, cup, orange) and press Enter: ").strip().lower()
        if user_input in model.names.values():
            target_object = user_input
            speak_mac(f"Looking for {target_object}")
            print(f"Tracking: {target_object} (Click back on the video window!)")
        else:
            print(f"'{user_input}' is not in the default list. Try: cell phone, bottle, cup, book, laptop, person, orange")
            speak_mac("Item not in list")
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
