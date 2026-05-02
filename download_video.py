"""
download_video.py — Download a public YouTube video for tracking.

Usage:
    python download_video.py --url "https://youtube.com/shorts/9LbamV50FTU?si=QTaqBcjy0UWvEnWT"
    python download_video.py --url "https://www.youtube.com/watch?v=XXXX" --output myvideo.mp4

Requires: yt-dlp  (pip install yt-dlp)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def download(url: str, output: str = "input_video.mp4"):
    output_path = Path(output)
    print(f"Downloading video from: {url}")
    print(f"Output path: {output_path.resolve()}")

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", str(output_path),
        "--no-playlist",
        url,
    ]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print("\n[ERROR] Download failed. Make sure yt-dlp is installed:")
        print("  pip install yt-dlp")
        sys.exit(1)

    print(f"\nDownload complete: {output_path.resolve()}")
    print(f"\nNow run the tracker:")
    print(f"  python tracker.py --video {output_path} --output output/output.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",    required=True, help="YouTube video URL")
    parser.add_argument("--output", default="input_video.mp4", help="Output filename")
    args = parser.parse_args()
    download(args.url, args.output)
