# Videos — Student Engagement Demo Clips

Place one video file per student here.

## Expected filenames

| Student | Primary filename | Alternatives accepted |
|---|---|---|
| Student 01 | `student01.mp4` | `student_01.mp4`, `student1.mp4` |
| Student 02 | `student02.mp4` | `student_02.mp4`, `student2.mp4` |
| Student 03 | `student03.mp4` | `student_03.mp4`, `student3.mp4` |
| Student 04 | `student04.mp4` | `student_04.mp4`, `student4.mp4` |
| Student 05 | `student05.mp4` | `student_05.mp4`, `student5.mp4` |

You only need as many files as you have "students".
The simulator will silently skip any slot without a matching file.

## Video requirements

| Property | Recommendation |
|---|---|
| Format | MP4 (H.264) — most common webcam export format |
| Resolution | 640 × 480 or higher |
| Frame rate | Any (processing is throttled to `--fps`, default 5 FPS) |
| Length | Any — videos loop by default (`--no-loop` disables this) |
| Content | **One clearly-visible frontal face per video** |

> **Note**: Each video must contain at most **one face**.
> The simulator runs a single-face MediaPipe FaceMesh per stream.
> If a frame contains multiple faces, only the first detected face is used.

## How to record suitable clips

### Option A — Use your webcam (recommended for demos)

```bash
# Record yourself sitting at a desk for 1–2 minutes.
# Do this for each "student" (or just re-use the same clip).
python demo/record_webcam.py --output videos/student01.mp4 --duration 120
```

### Option B — OBS / screen recorder

Open OBS → add a Webcam source → start recording → save as `student01.mp4`.

### Option C — Download a Creative-Commons talking-head clip

Any publicly available frontal face video works.
Suggested search terms: "talking head CC0 video", "webcam recording sample".

## Running the simulator

```bash
# Basic (headless, loops forever)
python demo/multi_student_sim.py

# With live OpenCV preview
python demo/multi_student_sim.py --display

# Custom video directory
python demo/multi_student_sim.py --video-dir path/to/my/videos

# Process each video once (no loop)
python demo/multi_student_sim.py --no-loop

# All options
python demo/multi_student_sim.py --video-dir videos/ --fps 5 --display --no-loop
```

## Viewing results

After starting the simulator, open the teacher dashboard in a second terminal:

```bash
.venv311\Scripts\streamlit run dashboard/app.py
```

Browse to `http://localhost:8501`.
The dashboard refreshes automatically every 5 seconds (configurable in the sidebar).

## Limitations

| Limitation | Reason |
|---|---|
| One face per video | Single-face FaceMesh per stream |
| No real-time webcam per student | Use `client/landmarks.py` for live multi-student |
| Gaze-on-screen always `True` | No calibration profile in demo mode |
| Blink count not separated from PERCLOS | Not surfaced by `DrowsinessMonitor.update()` |
| Emotion model requires face to be well-lit | Low-light clips may give poor emotion accuracy |
