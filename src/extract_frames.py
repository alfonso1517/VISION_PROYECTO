"""
Extract frames at 1 fps from tactical camera videos.
Skips videos below 720p minimum resolution.
Output: data/frames/<video_stem>/frame_XXXXXX.jpg

Use --start-sec / --end-sec to extract only the useful range and avoid
extracting thousands of frames that won't be used.
"""

import cv2
import argparse
from pathlib import Path

VIDEO_DIR  = Path("data/videos")
FRAMES_DIR = Path("data/frames")
MIN_HEIGHT = 720
JPG_QUALITY = 95
FPS_TARGET  = 1


def extract(video_path: Path, output_dir: Path,
            start_sec: int = 0, end_sec: int = 0,
            step: int = 1) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[SKIP] Cannot open: {video_path.name}")
        return 0

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if height < MIN_HEIGHT:
        print(f"[SKIP] Resolution too low ({height}p): {video_path.name}")
        cap.release()
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, round(src_fps / FPS_TARGET))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = start_sec * round(src_fps)
    end_frame   = (end_sec * round(src_fps)) if end_sec else total_frames

    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % frame_interval == 0:
            if saved % step == 0:
                out_path = output_dir / f"frame_{saved:06d}.jpg"
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])
            saved += 1
        frame_idx += 1

    cap.release()
    actual = saved // step + (1 if saved % step else 0)
    return saved


def main(video_dir: Path, frames_dir: Path,
         start_sec: int, end_sec: int, step: int,
         video_name: str) -> None:
    pattern = f"{video_name}.mp4" if video_name else "*.mp4"
    videos = sorted(video_dir.glob(pattern))
    if not videos:
        print(f"No .mp4 files found in {video_dir}")
        return

    for video_path in videos:
        out_dir = frames_dir / video_path.stem
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"[SKIP] Already extracted: {video_path.name}")
            continue
        print(f"Processing: {video_path.name} (sec {start_sec}→{end_sec or 'end'}, step={step})")
        n = extract(video_path, out_dir, start_sec, end_sec, step)
        if n:
            print(f"  Saved {n} frames → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames at 1 fps from tactical videos.")
    parser.add_argument("--video-dir",  type=Path, default=VIDEO_DIR)
    parser.add_argument("--frames-dir", type=Path, default=FRAMES_DIR)
    parser.add_argument("--video",      type=str,  default="", help="Nombre sin extensión (ej: tactico_02). Vacío = todos.")
    parser.add_argument("--start-sec",  type=int,  default=0,  help="Segundo de inicio")
    parser.add_argument("--end-sec",    type=int,  default=0,  help="Segundo de fin (0 = hasta el final)")
    parser.add_argument("--step",       type=int,  default=1,  help="Guardar 1 de cada N frames extraídos")
    args = parser.parse_args()
    main(args.video_dir, args.frames_dir, args.start_sec, args.end_sec, args.step, args.video)
