"""
Detección + tracking usando el modelo público de Roboflow Universe
football-players-detection-3zvbc/15 con ByteTrack (supervision).

Genera vídeo anotado y CSV con trayectorias de cada jugador.

Uso:
  python src/track_roboflow_universe.py
  python src/track_roboflow_universe.py --video tactico_02 --max-frames 300
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import supervision as sv
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

load_dotenv()

API_KEY   = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID  = "football-players-detection-3zvbc/15"
VIDEO_DIR = Path("data/videos")
CSV_OUT   = Path("outputs/metrics/tracks_universe.csv")
VIDEO_OUT = Path("outputs/tracked_videos")
CONF      = 0.5          # subido de 0.35 → reduce falsos positivos


def run(video_path: Path, csv_path: Path, video_out: Path,
        max_frames: int, start_sec: int) -> None:

    client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=API_KEY)

    # ByteTrack afinado para ~22 jugadores + árbitros
    tracker = sv.ByteTrack(
        track_activation_threshold=0.6,   # solo activa tracks con alta confianza
        lost_track_buffer=50,             # mantiene jugadores ocultos más tiempo
        minimum_matching_threshold=0.8,   # exige mejor coincidencia antes de reasignar ID
    )

    box_ann   = sv.BoxAnnotator(thickness=2)
    label_ann = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=3)

    cap = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_sec:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))

    limit = max_frames if max_frames else (total - int(start_sec * fps))

    video_out.parent.mkdir(parents=True, exist_ok=True)
    writer_video = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h),
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "track_id", "class", "x1", "y1", "x2", "y2", "confidence"])

    print(f"Modelo  : {MODEL_ID}")
    print(f"Vídeo   : {video_path.name} (desde seg {start_sec})")
    print(f"Frames  : {limit} | Conf mín: {CONF}")

    frame_idx      = 0
    total_conf     = 0.0
    total_dets     = 0
    unique_ids: set = set()

    while frame_idx < limit:
        ret, frame = cap.read()
        if not ret:
            break

        results    = client.infer(frame, model_id=MODEL_ID)
        detections = sv.Detections.from_inference(results)
        detections = detections[detections.confidence >= CONF]
        detections = tracker.update_with_detections(detections)

        if detections.tracker_id is not None:
            for i, tid in enumerate(detections.tracker_id):
                x1, y1, x2, y2 = detections.xyxy[i]
                conf  = float(detections.confidence[i])
                cls   = detections.data["class_name"][i] if "class_name" in detections.data else ""
                csv_writer.writerow([frame_idx, int(tid), cls,
                                     round(x1, 1), round(y1, 1),
                                     round(x2, 1), round(y2, 1),
                                     round(conf, 3)])
                unique_ids.add(int(tid))
                total_conf += conf
                total_dets += 1

        # Etiquetas: "clase #ID"
        labels = []
        if detections.tracker_id is not None:
            for cls_name, tid in zip(
                detections.data.get("class_name", []), detections.tracker_id
            ):
                labels.append(f"{cls_name} #{int(tid)}")

        annotated = box_ann.annotate(scene=frame.copy(), detections=detections)
        if labels:
            annotated = label_ann.annotate(scene=annotated, detections=detections, labels=labels)
        writer_video.write(annotated)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{limit} frames | IDs únicos: {len(unique_ids)}")

    cap.release()
    writer_video.release()
    csv_file.close()

    mean_conf = (total_conf / total_dets) if total_dets else 0
    print(f"\n{'='*45}")
    print(f"Frames procesados : {frame_idx}")
    print(f"Jugadores únicos  : {len(unique_ids)}")
    print(f"Confianza media   : {mean_conf:.3f}")
    print(f"Vídeo guardado en : {video_out}")
    print(f"CSV guardado en   : {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      type=str, default="tactico_02",
                        help="Nombre del vídeo sin extensión (en data/videos/)")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--start-sec",  type=int, default=165)
    args = parser.parse_args()

    video_path = VIDEO_DIR / f"{args.video}.mp4"
    video_out  = VIDEO_OUT / f"{args.video}_universe.mp4"
    run(video_path, CSV_OUT, video_out, args.max_frames, args.start_sec)
