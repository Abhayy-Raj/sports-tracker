"""
utils.py — Helper functions for the sports tracker pipeline.
Each function is standalone and testable independently.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from config import Config



# Logging


def setup_logger(name: str) -> logging.Logger:
    Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = Config.LOGS_DIR / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger

# Speed estimation


def estimate_speed(
    prev_bbox: np.ndarray,
    curr_bbox: np.ndarray,
    fps: float,
) -> float:
    """
    Estimate approximate speed in km/h from two consecutive bounding boxes.

    Uses the centre-bottom point (feet) to measure pixel displacement,
    then converts using Config.METERS_PER_PIXEL.

    Parameters
    ----------
    prev_bbox : [x1, y1, x2, y2] from previous frame
    curr_bbox : [x1, y1, x2, y2] from current frame
    fps       : frames per second of the video

    Returns
    -------
    speed_kmh : float
    """
    px1 = (prev_bbox[0] + prev_bbox[2]) / 2
    py1 = prev_bbox[3]
    px2 = (curr_bbox[0] + curr_bbox[2]) / 2
    py2 = curr_bbox[3]

    pixel_dist = np.hypot(px2 - px1, py2 - py1)
    meters     = pixel_dist * Config.METERS_PER_PIXEL
    speed_ms   = meters * fps              
    speed_kmh  = speed_ms * 3.6
    return min(speed_kmh, 60.0)             



# Team / jersey color assignment


def assign_team_by_color(
    frame: np.ndarray,
    bbox: np.ndarray,
) -> Optional[str]:
    """
    Assign a team label based on the dominant jersey color in the
    upper half of the bounding box (torso region).

    Uses HSV color ranges defined in Config.TEAM_HSV_RANGES.
    Returns the team name with the largest matching pixel area,
    or None if no team surpasses the minimum coverage threshold.

    Parameters
    ----------
    frame : BGR image (full frame)
    bbox  : [x1, y1, x2, y2]

    Returns
    -------
    team : str or None
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Clamp to frame bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return None

    
    torso_y2 = y1 + int((y2 - y1) * 0.55)
    roi = frame[y1:torso_y2, x1:x2]

    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    best_team = None
    best_ratio = 0.10  

    for team, (lo, hi) in Config.TEAM_HSV_RANGES.items():
        lower = np.array(lo, dtype=np.uint8)
        upper = np.array(hi, dtype=np.uint8)
        mask  = cv2.inRange(hsv, lower, upper)
        ratio = mask.sum() / (mask.size * 255 + 1e-6)
        if ratio > best_ratio:
            best_ratio = ratio
            best_team  = team

    return best_team



# Bird's-eye projection


def compute_bird_eye_point(
    bbox: np.ndarray,
    frame_w: int,
    frame_h: int,
    map_w: int,
    map_h: int,
) -> tuple[int, int]:
    """
    Project the centre-bottom of a bounding box onto a top-down map.

    Applies a simple perspective-like scaling: objects near the bottom
    of the frame (closer to the camera) map closer to the bottom of the
    bird's-eye view; objects near the top map near the top.

    Parameters
    ----------
    bbox    : [x1, y1, x2, y2]
    frame_w, frame_h : original frame dimensions
    map_w,   map_h   : bird's-eye canvas dimensions

    Returns
    -------
    (bx, by) : integer pixel coordinates on the map
    """
    cx = (bbox[0] + bbox[2]) / 2
    cy = bbox[3]

   
    nx = np.clip(cx / frame_w, 0, 1)
    ny = np.clip(cy / frame_h, 0, 1)

    bx = int(nx * (map_w - 1))
    by = int(ny * (map_h - 1))
    return bx, by



# Drawing utilities


def draw_trajectory(
    frame: np.ndarray,
    positions: list[tuple[int, int]],
    color: tuple[int, int, int],
    max_alpha: float = 0.8,
):
    """
    Draw a fading trajectory trail on the frame (in-place).

    Older positions are drawn with lower opacity, giving a
    motion-blur / comet-tail effect.

    Parameters
    ----------
    frame     : BGR image to draw on (modified in place)
    positions : list of (cx, cy) tuples, oldest first
    color     : BGR color for the trail
    max_alpha : maximum opacity for the most recent segment
    """
    n = len(positions)
    if n < 2:
        return

    overlay = frame.copy()

    for i in range(1, n):
        alpha = max_alpha * (i / n)
        thickness = max(1, int(3 * (i / n)))
        cv2.line(overlay, positions[i - 1], positions[i], color, thickness)

    cv2.addWeighted(overlay, max_alpha, frame, 1 - max_alpha, 0, frame)


def draw_speed_label(
    frame: np.ndarray,
    position: tuple[int, int],
    speed_kmh: float,
):
    """Draw a speed badge above a player's head."""
    x, y = position
    text  = f"{speed_kmh:.1f} km/h"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.rectangle(frame, (x - tw // 2 - 2, y - th - 8), (x + tw // 2 + 2, y - 2),
                  (0, 0, 0), -1)
    cv2.putText(frame, text, (x - tw // 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 180), 1, cv2.LINE_AA)


def draw_heatmap(accumulator: np.ndarray) -> np.ndarray:
    """
    Convert a float32 accumulation map into a coloured heatmap image.

    Parameters
    ----------
    accumulator : 2D float32 array of accumulated foot-positions

    Returns
    -------
    heatmap_img : BGR uint8 heatmap image
    """
    if accumulator.max() < 1:
        return np.zeros((*accumulator.shape, 3), dtype=np.uint8)


    blurred = cv2.GaussianBlur(accumulator, (51, 51), 0)


    norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)

    colormap = getattr(cv2, Config.HEATMAP_COLORMAP)
    heatmap  = cv2.applyColorMap(norm, colormap)
    return heatmap


def draw_player_count_chart(
    frame: np.ndarray,
    history: list[int],
):
    """
    Draw a small real-time line chart (bottom-left) showing active player
    count over the last N frames.

    Parameters
    ----------
    frame   : BGR image to draw on (modified in place)
    history : list of int counts (one per processed frame)
    """
    if len(history) < 2:
        return

    cw = Config.COUNT_CHART_WIDTH
    ch = Config.COUNT_CHART_HEIGHT
    margin_left   = 10
    margin_bottom = frame.shape[0] - ch - 10


    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (margin_left, margin_bottom),
        (margin_left + cw, margin_bottom + ch),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)


    cv2.rectangle(
        frame,
        (margin_left, margin_bottom),
        (margin_left + cw, margin_bottom + ch),
        (100, 100, 100),
        1,
    )

    cv2.putText(
        frame,
        "Players on screen",
        (margin_left + 4, margin_bottom + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA,
    )

    recent = history[-Config.COUNT_CHART_MAX_HISTORY:]
    if len(recent) < 2:
        return

    max_count = max(max(recent), 1)
    pad_x, pad_y = 8, 18
    plot_w = cw - 2 * pad_x
    plot_h = ch - pad_y - 8

    pts = []
    for i, cnt in enumerate(recent):
        px = margin_left + pad_x + int(i / (len(recent) - 1) * plot_w)
        py = margin_bottom + pad_y + plot_h - int(cnt / max_count * plot_h)
        pts.append((px, py))

    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], (0, 200, 255), 1)


    last_val = recent[-1]
    lx, ly = pts[-1]
    cv2.putText(
        frame,
        str(last_val),
        (lx + 2, ly),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1, cv2.LINE_AA,
    )
