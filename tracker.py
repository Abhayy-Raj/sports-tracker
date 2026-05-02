"""
Multi-Object Detection and Persistent ID Tracking Pipeline
===========================================================
Detects and tracks players/athletes in sports/event videos using
YOLOv8 + ByteTrack with extra features:
  - Trajectory trails
  - Movement heatmap
  - Player count over time chart
  - Speed estimation
  - Team clustering by jersey color
  - Top-view / bird's-eye projection
  - Per-player statistics CSV
  - Annotated output video
"""

import cv2
import numpy as np
import csv
import time
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import supervision as sv
from ultralytics import YOLO

from config import Config
from utils import (
    estimate_speed,
    assign_team_by_color,
    compute_bird_eye_point,
    draw_heatmap,
    draw_player_count_chart,
    draw_trajectory,
    setup_logger,
)

logger = setup_logger("tracker")


@dataclass
class PlayerState:
    """Holds per-player tracking state across frames."""
    track_id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=Config.TRAIL_LENGTH))
    speeds: deque = field(default_factory=lambda: deque(maxlen=30))
    team: Optional[str] = None
    total_distance: float = 0.0
    frame_count: int = 0
    last_bbox: Optional[np.ndarray] = None


class SportsTracker:
    """
    End-to-end pipeline:
      1. YOLOv8 detection
      2. ByteTrack ID assignment
      3. Feature extraction (speed, team, trajectory)
      4. Annotation & output
      5. Analytics export
    """

    def __init__(self, video_path: str, output_path: str):
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Create output directories
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        Config.SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("Loading YOLOv8 model: %s", Config.MODEL_NAME)
        self.model = YOLO(Config.MODEL_NAME)

        self.tracker = sv.ByteTrack(
            track_activation_threshold=Config.TRACK_ACTIVATION_THRESH,
            lost_track_buffer=Config.LOST_TRACK_BUFFER,
            minimum_matching_threshold=Config.MIN_MATCH_THRESH,
            frame_rate=Config.FRAME_RATE,
        )

        # FIX 1: Handle both old and new supervision versions
        try:
            self.box_annotator = sv.BoundingBoxAnnotator(thickness=2)
        except AttributeError:
            self.box_annotator = sv.RoundBoxAnnotator(thickness=2)

        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_padding=4,
        )

        self.players: dict = {}
        self.player_count_history: list = []
        self.heatmap_accumulator: Optional[np.ndarray] = None

        self._open_video()

    # ------------------------------------------------------------------
    # Video I/O
    # ------------------------------------------------------------------

    def _open_video(self):
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        self.frame_width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps          = self.cap.get(cv2.CAP_PROP_FPS) or Config.FRAME_RATE
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height),
        )

        self.heatmap_accumulator = np.zeros(
            (self.frame_height, self.frame_width), dtype=np.float32
        )

        logger.info(
            "Video: %dx%d @ %.1f fps | %d frames",
            self.frame_width, self.frame_height, self.fps, self.total_frames,
        )

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def _detect(self, frame: np.ndarray) -> sv.Detections:
        """Run YOLOv8 on a single frame, keep only person class."""
        results = self.model(
            frame,
            conf=Config.CONF_THRESHOLD,
            iou=Config.IOU_THRESHOLD,
            classes=[0],          
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        return detections

    def _get_active_ids(self, detections: sv.Detections) -> list:
        """FIX 2: Safely get list of active track IDs — avoids numpy bool ambiguity."""
        if detections.tracker_id is None:
            return []
        return [int(t) for t in detections.tracker_id if t is not None]

    def _update_player(
        self,
        track_id: int,
        bbox: np.ndarray,
        frame: np.ndarray,
        frame_idx: int,
    ) -> float:
        """Update or create PlayerState for a given track ID."""
        if track_id not in self.players:
            self.players[track_id] = PlayerState(track_id=track_id)

        player = self.players[track_id]

        # Centre-bottom point (feet position)
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int(bbox[3])
        player.positions.append((cx, cy))
        player.frame_count += 1

        # Speed estimation
        speed_kmh = 0.0
        if player.last_bbox is not None:
            speed_kmh = estimate_speed(player.last_bbox, bbox, self.fps)
            player.speeds.append(speed_kmh)
            prev_cx = int((player.last_bbox[0] + player.last_bbox[2]) / 2)
            prev_cy = int(player.last_bbox[3])
            player.total_distance += np.hypot(cx - prev_cx, cy - prev_cy)

        player.last_bbox = bbox.copy()

        # Team assignment — run once per player
        if player.team is None:
            player.team = assign_team_by_color(frame, bbox)

        # Heatmap update
        if 0 <= cy < self.frame_height and 0 <= cx < self.frame_width:
            self.heatmap_accumulator[cy, cx] += 1.0

        return speed_kmh

    # ------------------------------------------------------------------
    # Annotation helpers
    # ------------------------------------------------------------------

    def _build_labels(self, detections: sv.Detections, speed_map: dict) -> list:
        """Build display labels for each detection."""
        labels = []
        if detections.tracker_id is None:
            return ["?" for _ in detections.xyxy]

        for track_id in detections.tracker_id:
            if track_id is None:
                labels.append("?")
                continue
            player = self.players.get(int(track_id))
            team_tag = f" [{player.team}]" if (player and player.team) else ""
            spd = speed_map.get(int(track_id), 0.0)
            labels.append(f"ID {track_id}{team_tag} {spd:.1f}km/h")
        return labels

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        speed_map: dict,
        frame_idx: int,
        active_count: int,     
    ) -> np.ndarray:
        annotated = frame.copy()

       
        active_ids = self._get_active_ids(detections)
        for track_id in active_ids:
            player = self.players.get(track_id)
            if player and len(player.positions) > 1:
                color = Config.TEAM_COLORS.get(player.team, (0, 255, 0))
                draw_trajectory(annotated, list(player.positions), color)

        
        annotated = self.box_annotator.annotate(annotated, detections)

        
        labels = self._build_labels(detections, speed_map)
        annotated = self.label_annotator.annotate(annotated, detections, labels)

        
        self._draw_bird_eye_overlay(annotated, detections)

        draw_player_count_chart(annotated, self.player_count_history)

        
        cv2.putText(
            annotated,
            f"Frame {frame_idx}/{self.total_frames} | Active: {active_count}",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return annotated

    def _draw_bird_eye_overlay(self, frame: np.ndarray, detections: sv.Detections):
        """Render a small bird's-eye projection in the top-right corner."""
        margin = 10
        bw, bh = Config.BIRD_EYE_WIDTH, Config.BIRD_EYE_HEIGHT
        x0 = self.frame_width - bw - margin
        y0 = margin

        overlay = np.zeros((bh, bw, 3), dtype=np.uint8)
        cv2.rectangle(overlay, (0, 0), (bw - 1, bh - 1), (60, 60, 60), -1)
        cv2.rectangle(overlay, (0, 0), (bw - 1, bh - 1), (120, 120, 120), 1)

        # Simple pitch outline
        cv2.rectangle(overlay, (4, 4), (bw - 5, bh - 5), (40, 120, 40), 1)
        cv2.line(overlay, (bw // 2, 4), (bw // 2, bh - 5), (40, 120, 40), 1)

        if detections.tracker_id is not None:
            for bbox, track_id in zip(detections.xyxy, detections.tracker_id):
                if track_id is None:
                    continue
                bx, by = compute_bird_eye_point(
                    bbox, self.frame_width, self.frame_height, bw, bh
                )
                player = self.players.get(int(track_id))
                color = Config.TEAM_COLORS.get(
                    player.team if player else None, (0, 200, 200)
                )
                cv2.circle(overlay, (bx, by), 4, color, -1)
                cv2.putText(
                    overlay,
                    str(track_id),
                    (bx + 5, by + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        frame[y0:y0 + bh, x0:x0 + bw] = overlay


    # Main run loop
   

    def run(self):
        logger.info("Starting tracking pipeline...")
        start_time = time.time()
        frame_idx = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            
            if frame_idx % Config.FRAME_SKIP != 0:
                frame_idx += 1
                continue

            detections = self._detect(frame)
            detections = self.tracker.update_with_detections(detections)

            speed_map: dict = {}
            if detections.tracker_id is not None:
                for bbox, track_id in zip(detections.xyxy, detections.tracker_id):
                    if track_id is not None:
                        spd = self._update_player(int(track_id), bbox, frame, frame_idx)
                        speed_map[int(track_id)] = spd

            
            active_count = len(self._get_active_ids(detections))
            self.player_count_history.append(active_count)

            
            annotated = self._annotate_frame(
                frame, detections, speed_map, frame_idx, active_count
            )
            self.writer.write(annotated)

           
            if frame_idx % Config.SCREENSHOT_INTERVAL == 0:
                ss_path = Config.SCREENSHOTS_DIR / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(ss_path), annotated)
                logger.info("Screenshot saved: %s", ss_path)

            if frame_idx % 100 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    "Processed %d/%d frames (%.1fs) | Active players: %d",
                    frame_idx, self.total_frames, elapsed, active_count,
                )

            frame_idx += 1

        self._finalize()
        logger.info("Pipeline complete in %.1fs", time.time() - start_time)

   
    # Finalization
 
    def _finalize(self):
        self.cap.release()
        self.writer.release()
        logger.info("Output video saved: %s", self.output_path)

        self._export_heatmap()
        self._export_statistics()
        self._export_player_count_chart()

    def _export_heatmap(self):
        heatmap_img = draw_heatmap(self.heatmap_accumulator)
        path = Config.OUTPUT_DIR / "heatmap.jpg"
        cv2.imwrite(str(path), heatmap_img)
        logger.info("Heatmap saved: %s", path)

    def _export_statistics(self):
        path = Config.OUTPUT_DIR / "player_stats.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "track_id", "team", "frames_tracked",
                "avg_speed_kmh", "max_speed_kmh", "total_distance_px",
            ])
            for pid, p in sorted(self.players.items()):
                avg_spd = float(np.mean(p.speeds)) if p.speeds else 0.0
                max_spd = float(np.max(p.speeds)) if p.speeds else 0.0
                writer.writerow([
                    pid, p.team or "unknown", p.frame_count,
                    round(avg_spd, 2), round(max_spd, 2),
                    round(p.total_distance, 1),
                ])
        logger.info("Statistics saved: %s", path)

    def _export_player_count_chart(self):
        """Save standalone player count over time chart."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(self.player_count_history, color="#1f77b4", linewidth=1.2)
        ax.fill_between(
            range(len(self.player_count_history)),
            self.player_count_history,
            alpha=0.2,
            color="#1f77b4",
        )
        ax.set_xlabel("Frame (sampled)")
        ax.set_ylabel("Active players")
        ax.set_title("Player count over time")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = Config.OUTPUT_DIR / "player_count_chart.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        logger.info("Player count chart saved: %s", path)



# Entry point


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sports multi-object tracker")
    parser.add_argument("--video",  required=True, help="Path to input video")
    parser.add_argument("--output", default=str(Config.OUTPUT_DIR / "output.mp4"),
                        help="Path for annotated output video")
    args = parser.parse_args()

    pipeline = SportsTracker(args.video, args.output)
    pipeline.run()