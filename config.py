"""
config.py — Central configuration for the sports tracker pipeline.
Edit values here; no need to touch tracker.py or utils.py.
"""

from pathlib import Path


class Config:
   
    BASE_DIR        = Path(__file__).parent
    OUTPUT_DIR      = BASE_DIR / "output"
    SCREENSHOTS_DIR = BASE_DIR / "screenshots"
    LOGS_DIR        = BASE_DIR / "logs"

   
    MODEL_NAME = "yolov8m.pt"


    CONF_THRESHOLD = 0.50


    IOU_THRESHOLD = 0.45

    
    TRACK_ACTIVATION_THRESH = 0.50   
    LOST_TRACK_BUFFER       = 30     
    MIN_MATCH_THRESH        = 0.60  
    FRAME_RATE              = 30     

   
    FRAME_SKIP = 1

   
    SCREENSHOT_INTERVAL = 300

   
    TRAIL_LENGTH = 25

   
    BIRD_EYE_WIDTH  = 160
    BIRD_EYE_HEIGHT = 120

   
    COUNT_CHART_WIDTH  = 260
    COUNT_CHART_HEIGHT = 80
    COUNT_CHART_MAX_HISTORY = 200  

    HEATMAP_COLORMAP = "COLORMAP_JET"

   
    METERS_PER_PIXEL = 0.05


    
    
    
    TEAM_COLORS = {
        "team_A": (0,   200, 255),   
        "team_B": (255, 100,  50),   
        "referee":(200, 200, 200),   
        None:     (0,   255,   0),   
    }


   

    TEAM_HSV_RANGES = {
        "team_A":  ([0,   120, 70],  [10,  255, 255]),   # red jerseys
        "team_B":  ([100, 80,  80],  [130, 255, 255]),   # blue jerseys
        "referee": ([0,   0,   180], [180, 30,  255]),   # white/yellow
    }