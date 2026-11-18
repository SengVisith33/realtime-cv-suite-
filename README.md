# Mac Computer Vision Playground

Real-time AI computer vision apps built with Python, OpenCV, YOLOv8, and MediaPipe.
cd ~/object-finder && cat << 'EOF' > README.md
# Real-Time Computer Vision Suite

A collection of low-latency, local computer vision and spatial interaction applications built with **Python**, **OpenCV**, **YOLOv8**, and **MediaPipe**. Designed to run entirely on-device without cloud inference latency.

## Projects Included

| File | Application | Core Tech & Mechanics |
| :--- | :--- | :--- |
| `finder.py` | **Assistive Object Locator** | Uses **YOLOv8** bounding-box detection and horizontal frame-offset math to calculate spatial position (`LEFT`, `CENTERED`, `RIGHT`) with asynchronous macOS speech feedback. |
| `hands.py` | **21-Point Hand Landmarker** | Tracks multi-hand skeletal geometry in real time using the **MediaPipe Tasks API** and computes finger-joint states for live gesture counting. |
| `ninja.py` | **AR Gesture Arcade** | Maps index-fingertip velocity trails to 2D physics collision detection for an interactive, touchless slicing mini-game. |
| `canvas.py` | **Spatial Air Canvas** | Converts finger-joint states into a state machine (1 finger = draw, 2 fingers = cycle palette) to render persistent overlays onto a live video stream. |

## Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd realtime-cv-suite
   ```

2. **Install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run any module:**
   ```bash
   python finder.py
   python hands.py
   python ninja.py
   python canvas.py
   ```
EOF
