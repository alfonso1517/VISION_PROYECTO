"""
Detección + tracking sobre tactico_01.mp4 usando el modelo Roboflow
football-tactical/2 y ByteTrack (supervision).

Uso: python src/track.py [--input PATH] [--output PATH] [--conf FLOAT]
"""

import argparse
import os
from pathlib import Path

import cv2
import supervision as sv
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

load_dotenv()

API_KEY    = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID   = "football-tactical/2"
INPUT_VIDEO  = Path("data/videos/tactico_01.mp4")
OUTPUT_DIR   = Path("outputs/tracked_videos")
OUTPUT_VIDEO = OUTPUT_DIR / "tactico_01_tracked.mp4"
CONF_THRESH  = 0.35

# Colores por clase: player, goalkeeper, referee, ball
CLASS_COLORS = {
    "player":     sv.Color.from_hex("#00B7EB"),
    "goalkeeper": sv.Color.from_hex("#00FFCE"),
    "referee":    sv.Color.from_hex("#FFFF00"),
    "ball":       sv.Color.from_hex("#FF8000"),
}


def build_annotators() -> tuple:
    box = sv.BoxAnnotator(thickness=2)
    label = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=3)
    return box, label


def run(input_path: Path, output_path: Path, conf: float,
        max_frames: int = 0, start_sec: int = 0) -> None:
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=API_KEY,
    )
    tracker = sv.ByteTrack()
    box_ann, label_ann = build_annotators()

    cap = cv2.VideoCapture(str(input_path))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_sec:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    if max_frames:
        total = max_frames
    print(f"Procesando {total} frames desde seg {start_sec} → {output_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inferencia en Roboflow cloud
        results = client.infer(frame, model_id=MODEL_ID)
        detections = sv.Detections.from_inference(results)

        # Filtrar por confianza
        detections = detections[detections.confidence >= conf]

        # ByteTrack
        detections = tracker.update_with_detections(detections)

        # Etiquetas: "clase #ID"
        labels = []
        if detections.tracker_id is not None:
            for cls_name, tid in zip(detections.data.get("class_name", []), detections.tracker_id):
                labels.append(f"{cls_name} #{tid}" if tid is not None else str(cls_name))

        annotated = box_ann.annotate(scene=frame.copy(), detections=detections)
        if labels:
            annotated = label_ann.annotate(scene=annotated, detections=detections, labels=labels)

        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total} frames procesados")
        if max_frames and frame_idx >= max_frames:
            break

    cap.release()
    writer.release()
    print(f"\nVídeo guardado en {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=Path, default=INPUT_VIDEO)
    parser.add_argument("--output", type=Path, default=OUTPUT_VIDEO)
    parser.add_argument("--conf",       type=float, default=CONF_THRESH)
    parser.add_argument("--max-frames", type=int,   default=0)
    parser.add_argument("--start-sec",  type=int,   default=0)
    args = parser.parse_args()
    run(args.input, args.output, args.conf, args.max_frames, args.start_sec)
