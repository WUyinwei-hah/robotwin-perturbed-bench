#!/usr/bin/env python3
"""
Combine real environment videos and model predicted videos side by side.

Usage:
    python combine_videos.py --results_dir <path_to_eval_results>

This script finds pairs of episodeN.mp4 (real env) and episodeN_pred.mp4 (model predicted)
and combines them into episodeN_combined.mp4 with labels.
"""

import os
import re
import subprocess
import argparse
from pathlib import Path


def find_video_pairs(results_dir):
    """Find all (real, pred) video pairs recursively."""
    results_dir = Path(results_dir)
    pairs = []

    for real_video in sorted(results_dir.rglob("episode*.mp4")):
        name = real_video.name
        # Skip pred and combined videos
        if "_pred" in name or "_combined" in name:
            continue
        # Match episodeN.mp4
        match = re.match(r"(episode\d+)\.mp4", name)
        if not match:
            continue
        prefix = match.group(1)
        pred_video = real_video.parent / f"{prefix}_pred.mp4"
        combined_video = real_video.parent / f"{prefix}_combined.mp4"
        if pred_video.exists():
            pairs.append((str(real_video), str(pred_video), str(combined_video)))
        else:
            print(f"  Warning: no predicted video for {real_video}")

    return pairs


def combine_side_by_side(real_path, pred_path, output_path, target_height=384):
    """
    Combine real env video (left) and predicted video (right) side by side.
    Both are scaled to the same height. Labels are added on top.
    """
    # Use ffmpeg filter_complex to:
    # 1. Scale both to same height
    # 2. Add text labels
    # 3. Stack horizontally
    filter_complex = (
        f"[0:v]scale=-2:{target_height},drawtext=text='Real Environment':"
        f"fontsize=20:fontcolor=white:borderw=2:bordercolor=black:x=(w-tw)/2:y=10[left];"
        f"[1:v]scale=-2:{target_height},drawtext=text='Model Predicted':"
        f"fontsize=20:fontcolor=white:borderw=2:bordercolor=black:x=(w-tw)/2:y=10[right];"
        f"[left][right]hstack=inputs=2[out]"
    )

    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-i", real_path,
        "-i", pred_path,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-shortest",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg error: {result.stderr.strip()}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Combine real and predicted videos side by side")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Root directory containing eval results with video pairs")
    parser.add_argument("--height", type=int, default=384,
                        help="Target height for combined video (default: 384)")
    args = parser.parse_args()

    print(f"Scanning for video pairs in: {args.results_dir}")
    pairs = find_video_pairs(args.results_dir)
    print(f"Found {len(pairs)} video pairs\n")

    success = 0
    for real_path, pred_path, output_path in pairs:
        rel_path = os.path.relpath(output_path, args.results_dir)
        print(f"  Combining: {rel_path}")
        if combine_side_by_side(real_path, pred_path, output_path, args.height):
            success += 1
        else:
            print(f"    FAILED: {rel_path}")

    print(f"\nDone: {success}/{len(pairs)} combined successfully")


if __name__ == "__main__":
    main()
