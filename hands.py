import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2, min_hand_detection_confidence=0.6)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
TIP_IDS = (8, 12, 16, 20)
CONNECTIONS = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17))

print("READY! Hold your hand up to the camera. Press Q on the video window to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    fingers_up = 0

    if result.hand_landmarks:
        for lms, handedness in zip(result.hand_landmarks, result.handedness):
            pts = tuple(lms)
            for a, b in CONNECTIONS:
                pt1 = (int(pts[a].x * w), int(pts[a].y * h))
                pt2 = (int(pts[b].x * w), int(pts[b].y * h))
                cv2.line(frame, pt1, pt2, (255, 255, 255), 2)
            for lm in pts:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 0, 255), -1)

            first_hand = next(iter(handedness))
            thumb_tip = pts[4]
            thumb_joint = pts[3]
            if first_hand.category_name == "Right":
                if thumb_tip.x < thumb_joint.x:
                    fingers_up += 1
            else:
                if thumb_tip.x > thumb_joint.x:
                    fingers_up += 1

            for tip in TIP_IDS:
                if pts[tip].y < pts[tip - 2].y:
                    fingers_up += 1

    cv2.putText(frame, f"Fingers Up: {fingers_up}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("5-Finger Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
