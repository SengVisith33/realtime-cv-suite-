import cv2
import random
import math
import subprocess
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, min_hand_detection_confidence=0.6)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
trail = []
items = []
score = 0
game_over = False

def spawn_item(w, h):
    is_bomb = random.random() < 0.25
    colors = ((0, 0, 255), (0, 255, 0), (0, 165, 255), (255, 0, 255), (0, 255, 255))
    return {
        "x": float(random.randint(100, w - 100)),
        "y": float(h + 30),
        "vx": random.uniform(-4.0, 4.0),
        "vy": random.uniform(-24.0, -17.0),
        "color": (20, 20, 20) if is_bomb else random.choice(colors),
        "r": 34 if is_bomb else 30,
        "bomb": is_bomb
    }

print("READY! Slice fruits, dodge the black bombs! Press R to restart or Q to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    if not game_over:
        if random.random() < 0.05:
            items.append(spawn_item(w, h))

        finger_pt = None
        if result.hand_landmarks:
            hand = next(iter(result.hand_landmarks))
            tip = list(hand).pop(8)
            finger_pt = (int(tip.x * w), int(tip.y * h))
            trail.append(finger_pt)
            if len(trail) > 12:
                trail.pop(0)
        else:
            trail.clear()

        for i in range(1, len(trail)):
            cv2.line(frame, list(trail).pop(i - 1), list(trail).pop(i), (255, 255, 255), i * 2)

        active_items = []
        for item in items:
            item["x"] += item["vx"]
            item["y"] += item["vy"]
            item["vy"] += 0.65

            hit = False
            if finger_pt:
                fx, fy = finger_pt
                if math.hypot(fx - item["x"], fy - item["y"]) < item["r"] + 16:
                    hit = True
                    if item["bomb"]:
                        game_over = True
                        subprocess.Popen(["say", "Boom! Game over!"])
                        cv2.circle(frame, (int(item["x"]), int(item["y"])), 120, (0, 0, 255), -1)
                    else:
                        score += 1
                        cv2.circle(frame, (int(item["x"]), int(item["y"])), 48, (255, 255, 255), 4)

            if not hit and item["y"] < h + 60:
                pos = (int(item["x"]), int(item["y"]))
                cv2.circle(frame, pos, item["r"], item["color"], -1)
                if item["bomb"]:
                    cv2.circle(frame, pos, item["r"], (0, 0, 255), 3)
                    cv2.putText(frame, "BOMB", (pos[0] - 28, pos[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                active_items.append(item)
        items = active_items

    cv2.putText(frame, f"Score: {score}", (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
    if game_over:
        cv2.putText(frame, "GAME OVER! Press R to Restart", (w // 2 - 260, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    cv2.imshow("Webcam Fruit Ninja", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r") and game_over:
        items.clear()
        trail.clear()
        score = 0
        game_over = False

cap.release()
cv2.destroyAllWindows()
