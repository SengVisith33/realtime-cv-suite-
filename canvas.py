import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, min_hand_detection_confidence=0.6)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
canvas = None
prev_pt = None
color_idx = 0
colors = ((0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 150, 0), (0, 0, 255))
names = ("GREEN", "PINK", "YELLOW", "BLUE", "RED")
cooldown = 0

print("READY! 1 finger = Draw | 2 fingers = Change Color | Press C = Clear | Q = Quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    if canvas is None:
        canvas = np.zeros_like(frame)

    if cooldown > 0:
        cooldown -= 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    if result.hand_landmarks:
        pts = tuple(next(iter(result.hand_landmarks)))
        idx_tip = pts[8]
        idx_pip = pts[6]
        mid_tip = pts[12]
        mid_pip = pts[10]

        index_up = idx_tip.y < idx_pip.y
        middle_up = mid_tip.y < mid_pip.y
        curr_pt = (int(idx_tip.x * w), int(idx_tip.y * h))

        if index_up and middle_up:
            prev_pt = None
            cv2.circle(frame, curr_pt, 18, (255, 255, 255), 3)
            if cooldown == 0:
                color_idx = (color_idx + 1) % len(colors)
                cooldown = 25
        elif index_up:
            curr_color = colors[color_idx]
            cv2.circle(frame, curr_pt, 10, curr_color, -1)
            if prev_pt is not None:
                cv2.line(canvas, prev_pt, curr_pt, curr_color, 8)
            prev_pt = curr_pt
        else:
            prev_pt = None
    else:
        prev_pt = None

    combined = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)
    curr_color = colors[color_idx]
    cv2.putText(combined, f"Color: {names[color_idx]} (2 Fingers = Switch | C = Clear | Q = Quit)", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, curr_color, 2)
    cv2.imshow("Air Canvas", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()
