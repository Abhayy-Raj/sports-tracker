# Technical Report: Multi-Object Detection and Persistent ID Tracking

**Assignment:** AI / Computer Vision / Data Science  
**Duration:** 2–3 days  
**Video Source:** *(add your YouTube link here)*

---

## 1. Overview

This project implements a real-time multi-object tracking pipeline for sports and public event footage. The system detects all players/athletes present in a video, assigns each a persistent unique ID, and produces an annotated output video alongside analytics exports (heatmap, speed statistics, player count chart, bird's-eye view, team clustering).

---

## 2. Model and Detector: YOLOv8

**Model chosen:** `yolov8m.pt` (medium variant, COCO-pretrained)

YOLOv8 (You Only Look Once, version 8) by Ultralytics is a single-stage object detector that processes the entire image in one forward pass. It predicts bounding boxes and class probabilities directly, making it highly suited to real-time video inference.

**Why YOLOv8:**
- Excellent trade-off between speed and accuracy on the COCO person class
- Pre-trained weights generalise well to sports footage without fine-tuning
- Native integration with the `supervision` library simplifies annotation
- `yolov8m` provides strong detection at 720p without requiring a GPU cluster

**Configuration:**
- Confidence threshold: `0.40` (filters low-confidence ghost detections)
- IoU (NMS) threshold: `0.45` (reduces duplicate boxes in crowded scenes)
- Class filter: `person` only (class index 0 in COCO)

---

## 3. Tracking Algorithm: ByteTrack

**Tracker chosen:** ByteTrack (via `supervision.ByteTrack`)

ByteTrack (Zhang et al., 2022) is a multi-object tracking algorithm that matches detected boxes to existing tracks using IoU-based assignment. Its key innovation is including *low-confidence* detections in the matching step — this recovers IDs for partially occluded players that a naive high-threshold tracker would drop.

**Why ByteTrack:**
- State-of-the-art ID consistency under occlusion
- No appearance model required (no re-ID network), which keeps latency low
- Works well with imperfect detectors in crowded sports scenes
- Lightweight: no GPU memory overhead beyond the detector

**Parameters used:**
| Parameter | Value | Reason |
|---|---|---|
| `track_activation_threshold` | 0.25 | Allows recovering low-confidence players |
| `lost_track_buffer` | 60 frames | Keeps ID alive during 2-second occlusions at 30fps |
| `minimum_matching_threshold` | 0.80 | High IoU requirement prevents wrong ID merges |

---

## 4. How ID Consistency Is Maintained

ID consistency is maintained through three layers:

1. **IoU-based matching:** ByteTrack matches each detected box to the closest predicted track location using intersection-over-union. Players who move predictably between frames are matched with high confidence.

2. **Kalman filter motion prediction:** ByteTrack internally maintains a Kalman filter per track, predicting where each player will be in the next frame. This bridges short gaps where detection fails (e.g., blur or occlusion).

3. **Lost-track buffer:** When a player is fully occluded, their track is not immediately deleted. ByteTrack holds it in a "lost" state for `lost_track_buffer` frames. If the player reappears within that window, the same ID is restored.

---

## 5. Extra Features Implemented

| Feature | Description |
|---|---|
| **Trajectory trails** | Fading comet-tail drawn from each player's last 40 positions |
| **Speed estimation** | Approximate km/h from frame-to-frame displacement scaled by `METERS_PER_PIXEL` |
| **Team clustering** | HSV jersey-color heuristic assigns team_A / team_B / referee labels |
| **Bird's-eye mini-map** | Top-right overlay projecting all players onto a top-down pitch view |
| **Player count chart** | Bottom-left real-time line chart of active player count |
| **Heatmap** | Accumulated foot-position density map exported as `heatmap.jpg` |
| **Statistics CSV** | Per-player: avg speed, max speed, frames tracked, total distance |
| **Player count chart (export)** | High-resolution PNG chart saved to `output/` |

---

## 6. Challenges Faced

**Occlusion:** Players frequently overlap in tight formations. ByteTrack's lost-track buffer mitigates most cases, but ID switches still occur during prolonged full-body occlusions.

**Camera motion:** Panning and zooming cameras cause large pixel displacements between frames, reducing IoU matching reliability. Reducing `lost_track_buffer` would worsen this; the current setting of 60 is a compromise.

**Similar appearance:** Players from the same team wearing identical jerseys have no distinguishing visual features. Without a dedicated re-ID network, ID recovery after full occlusion relies entirely on spatial proximity.

**Scale changes:** Zoomed-out shots reduce bounding box size, dropping detection confidence. The `yolov8m` model handles this better than `yolov8n` at the cost of inference time.

**Speed estimation accuracy:** Pixel-to-meter conversion is a rough approximation. Accurate speed requires camera calibration or known ground-plane homography. The values provided are indicative, not ground-truth.

---

## 7. Failure Cases Observed

- **ID switch after full occlusion by another player** — most common failure; occurs when two players cross and emerge in swapped positions
- **Missed detections in motion blur** — fast sprinting causes blur that drops confidence below threshold; the subsequent re-detection gets a new ID
- **Referee misclassified** — referees in white/black can be matched to either team depending on lighting
- **Ghost tracks at frame edges** — partial detections at the video border occasionally persist for several frames before being dropped

---

## 8. Possible Improvements

1. **Add a re-ID appearance model** (e.g., StrongSORT or BoT-SORT) — associates track IDs using visual embeddings, drastically reducing ID switches after full occlusion

2. **Homography-based ground-plane calibration** — use known pitch dimensions to convert pixel coordinates to real-world metres, enabling accurate speed and distance measurement

3. **Fine-tune detector on sports data** — training YOLOv8 on a sports-specific dataset (e.g., SoccerNet) would improve recall on small/occluded players

4. **Temporal smoothing of team labels** — majority-vote over a sliding window would eliminate flickering team assignments under varying lighting

5. **Multi-camera fusion** — for broadcast footage with multiple camera angles, fusing detections across cameras would eliminate blind spots

6. **Evaluation metrics** — implementing HOTA, MOTA, and IDF1 against a labelled ground truth would quantify tracking quality objectively

---

## 9. Model and Library Versions

| Component | Version |
|---|---|
| Python | 3.10+ |
| ultralytics (YOLOv8) | ≥ 8.2.0 |
| supervision (ByteTrack) | ≥ 0.21.0 |
| OpenCV | ≥ 4.9.0 |
| NumPy | ≥ 1.26.0 |
| Matplotlib | ≥ 3.8.0 |

---

## 10. References

- Jocher, G. et al. (2023). *Ultralytics YOLOv8*. GitHub.
- Zhang, Y. et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. ECCV 2022.
- Bewley, A. et al. (2016). *Simple Online and Realtime Tracking*. ICIP 2016.
- Supervision library: https://github.com/roboflow/supervision
