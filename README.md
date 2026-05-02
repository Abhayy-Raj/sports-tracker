# Multi-Object Detection and Persistent ID Tracking

> **Assignment submission** — AI / Computer Vision / Data Science  
> Video source: https://youtube.com/shorts/9LbamV50FTU?si=QTaqBcjy0UWvEnWT

---

## What this does

This pipeline detects every player/athlete in a sports video and assigns each one a stable unique ID that persists across the entire video — even through occlusion, camera motion, and fast movement.

**Features:**
- YOLOv8 person detection (no training required)
- ByteTrack multi-object tracking with persistent IDs
- Fading trajectory trails per player
- Approximate speed estimation (km/h)
- Team/role assignment by jersey color (HSV heuristic)
- Real-time bird's-eye mini-map overlay
- Live player count chart overlay
- Movement heatmap export
- Per-player statistics CSV
- Annotated output video

---

## Project structure

```
sports_tracker/
├── tracker.py            # Main pipeline (detection + tracking + annotation)
├── utils.py              # Helper functions (speed, heatmap, drawing, etc.)
├── config.py             # All tunable parameters in one place
├── download_video.py     # Optional: download a YouTube video
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── TECHNICAL_REPORT.md   # 2-page technical report
├── output/
│   ├── output.mp4        # Annotated output video (generated)
│   ├── heatmap.jpg       # Movement heatmap (generated)
│   ├── player_stats.csv  # Per-player stats (generated)
│   └── player_count_chart.png  # Player count over time (generated)
├── screenshots/          # Auto-saved frame screenshots
└── logs/                 # Log files
```

---

## Installation

### 1. Clone / unzip the project

```bash
cd sports_tracker
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

YOLOv8 weights (`yolov8m.pt`) download automatically on first run.

---

## Usage

### Option A — You already have a video file

```bash
python tracker.py --video path/to/your_video.mp4 --output output/output.mp4
```

### Option B — Download from YouTube first

```bash
# Install yt-dlp if not already installed
pip install yt-dlp

# Download the video
python download_video.py --url "https://youtube.com/shorts/9LbamV50FTU?si=QTaqBcjy0UWvEnWT"

# Then run the tracker
python tracker.py --video input_video.mp4 --output output/output.mp4
```

### Outputs

After the pipeline finishes you will find:

| File | Description |
|---|---|
| `output/output.mp4` | Annotated video with bounding boxes, IDs, trajectories |
| `output/heatmap.jpg` | Movement density heatmap across the full video |
| `output/player_stats.csv` | Per-player speed, distance, team, frames tracked |
| `output/player_count_chart.png` | High-res chart of active player count over time |
| `screenshots/*.jpg` | Sampled frames saved every 300 processed frames |

---

## Configuration

All parameters are in `config.py`. Key settings:

| Parameter | Default | Effect |
|---|---|---|
| `MODEL_NAME` | `yolov8m.pt` | Swap to `yolov8n.pt` for speed or `yolov8x.pt` for accuracy |
| `CONF_THRESHOLD` | `0.40` | Lower → more detections (more false positives) |
| `LOST_TRACK_BUFFER` | `60` | Higher → longer ID retention during occlusion |
| `FRAME_SKIP` | `1` | Set to `2` or `3` to process faster on slow machines |
| `TRAIL_LENGTH` | `40` | Number of past positions in trajectory trail |
| `METERS_PER_PIXEL` | `0.05` | Adjust for your video's camera height/zoom |
| `TEAM_HSV_RANGES` | (see config) | Tune hue ranges to match jersey colors in your video |

---

## Assumptions

- Video contains human subjects (players/athletes/participants)
- Input video is a standard MP4/AVI/MOV file readable by OpenCV
- `METERS_PER_PIXEL = 0.05` is a rough estimate; speed values are indicative
- Team color clustering works best when two teams wear clearly distinct colors
- GPU is optional — the pipeline runs on CPU (slower) and CUDA GPU (faster)

---

## Limitations

- **ID switches** can occur when two players fully occlude each other and emerge in swapped positions
- **Speed values** are approximate; accurate measurement requires camera calibration
- **Team assignment** is a heuristic and may misclassify under poor lighting or unusual jersey colors
- **Fast camera pans** reduce matching reliability due to large inter-frame displacement
- No re-ID appearance model is used; tracking relies entirely on spatial overlap (IoU)

---

## Hardware requirements

| Setup | Expected speed |
|---|---|
| CPU only | 2–5 fps (usable for offline processing) |
| NVIDIA GPU (CUDA) | 20–60 fps depending on model size |

For GPU acceleration, ensure `torch` with CUDA is installed:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `ultralytics` | YOLOv8 detection |
| `supervision` | ByteTrack + annotation helpers |
| `opencv-python` | Video I/O + drawing |
| `numpy` | Array math |
| `matplotlib` | Chart export |
| `yt-dlp` | YouTube video download (optional) |
