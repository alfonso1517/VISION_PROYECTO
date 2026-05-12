"""
Detección + tracking sobre vídeo táctico usando el modelo Roboflow
football-tactical/2 y ByteTrack (supervision).

Incluye:
- Interpolación de posición del balón (abdullahtarek/football-analysis)
- Clasificación de equipo: SigLIP → UMAP → KMeans, cacheado por track_id

Uso: python src/track.py [--input PATH] [--output PATH] [--conf FLOAT]
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

sys.path.insert(0, str(Path(__file__).parent))
from team_assigner import TeamClassifier

load_dotenv()

API_KEY      = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID     = "football-tactical/4"
INPUT_VIDEO  = Path("data/videos/tactico_01.mp4")
OUTPUT_DIR   = Path("outputs/tracked_videos")
OUTPUT_VIDEO = OUTPUT_DIR / "tactico_01_tracked.mp4"
CONF_THRESH       = 0.35   # jugadores, árbitros, porteros
CONF_THRESH_BALL  = 0.05   # balón: umbral muy agresivo para no perdérselo

# Paleta: class_id 0→equipo A (verde), 1→equipo B (rojo), 2→árbitro/portero (amarillo)
COLOR_PALETTE = sv.ColorPalette.from_hex(["#00FF00", "#FF4444", "#FFFF00"])
BALL_COLOR    = sv.Color.from_hex("#FFFFFF")


def interpolate_ball(ball_bboxes: list) -> list:
    """Rellena posiciones de balón faltantes con interpolación lineal."""
    df = pd.DataFrame(
        [b if b is not None else [np.nan] * 4 for b in ball_bboxes],
        columns=["x1", "y1", "x2", "y2"],
    )
    df = df.interpolate().bfill()
    result = []
    for _, row in df.iterrows():
        result.append(None if np.isnan(row["x1"]) else row.tolist())
    return result


def run(input_path: Path, output_path: Path, conf: float,
        max_frames: int = 0, start_sec: int = 0) -> None:

    client     = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=API_KEY)
    tracker    = sv.ByteTrack()
    classifier = TeamClassifier()   # detecta CUDA automáticamente con torch

    cap   = cv2.VideoCapture(str(input_path))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_sec:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))

    limit = max_frames if max_frames else (total - int(start_sec * fps))
    print(f"Pasada 1/2 — inferencia API ({limit} frames desde seg {start_sec})...")

    frames_buf: list[np.ndarray]        = []
    detections_buf: list[sv.Detections] = []
    team_buf: list[dict[int, int]]      = []   # {track_id: team_id} por frame
    ball_bboxes: list                   = []
    fitted = False

    frame_idx = 0
    while frame_idx < limit:
        ret, frame = cap.read()
        if not ret:
            break

        results    = client.infer(frame, model_id=MODEL_ID)
        detections = sv.Detections.from_inference(results)

        # Threshold diferenciado: balón más permisivo para no perdérselo
        raw_classes = detections.data.get("class_name", np.array([]))
        ball_mask   = np.array([c == "ball" for c in raw_classes])
        conf_mask   = (
            (~ball_mask & (detections.confidence >= conf)) |
            ( ball_mask & (detections.confidence >= CONF_THRESH_BALL))
        )
        detections = detections[conf_mask]
        detections = tracker.update_with_detections(detections)

        classes = detections.data.get("class_name", np.array([]))

        # Fit del clasificador en el primer frame con ≥4 jugadores
        if not fitted:
            player_mask = np.array([c == "player" for c in classes])
            if player_mask.sum() >= 4:
                crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy[player_mask]]
                classifier.fit(crops)
                fitted = True

        # Asignación de equipo con restricción de cardinalidad (máx 13 por equipo/frame)
        frame_teams: dict[int, int] = {}
        frame_counts: dict[int, int] = {0: 0, 1: 0}
        if detections.tracker_id is not None:
            for j, (cls, tid) in enumerate(zip(classes, detections.tracker_id)):
                if cls == "player":
                    crop    = sv.crop_image(frame, detections.xyxy[j])
                    team_id = classifier.get_team_with_limit(crop, int(tid), frame_counts)
                    frame_teams[int(tid)] = team_id
                    frame_counts[team_id] = frame_counts.get(team_id, 0) + 1

        # Balón para interpolación
        ball_bbox = None
        for j, cls in enumerate(classes):
            if cls == "ball":
                ball_bbox = detections.xyxy[j].tolist()
                break
        ball_bboxes.append(ball_bbox)

        frames_buf.append(frame)
        detections_buf.append(detections)
        team_buf.append(frame_teams)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{limit} frames | IDs cacheados: {len(classifier._cache)}")

    cap.release()

    # Interpolación balón
    print("Interpolando posición del balón...")
    ball_interp   = interpolate_ball(ball_bboxes)
    ball_detected = sum(1 for b in ball_bboxes if b is not None)
    ball_filled   = sum(1 for b in ball_interp if b is not None) - ball_detected

    # Pasada 2: renderizado
    print(f"Pasada 2/2 — renderizando → {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    ellipse_ann = sv.EllipseAnnotator(color=COLOR_PALETTE, thickness=2)
    label_ann   = sv.LabelAnnotator(
        color=COLOR_PALETTE,
        text_color=sv.Color.BLACK,
        text_scale=0.45,
        text_thickness=1,
        text_padding=3,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    triangle_ann = sv.TriangleAnnotator(
        color=BALL_COLOR, base=20, height=17, outline_thickness=1
    )

    team_counts: dict[int, set] = defaultdict(set)

    for i, (frame, detections) in enumerate(zip(frames_buf, detections_buf)):
        # Aplica balón interpolado
        ball_bbox = ball_interp[i]
        if ball_bbox is not None and "class_name" in detections.data:
            for j, cls in enumerate(detections.data["class_name"]):
                if cls == "ball":
                    detections.xyxy[j] = np.array(ball_bbox)
                    break

        classes     = detections.data.get("class_name", np.array([]))
        frame_teams = team_buf[i]
        n           = len(detections)
        new_class_ids = np.zeros(n, dtype=int)
        labels = []

        for j, cls in enumerate(classes):
            tid = int(detections.tracker_id[j]) if detections.tracker_id is not None else -1
            if cls == "player":
                team_id = frame_teams.get(tid, 0)
                new_class_ids[j] = team_id          # 0→verde, 1→rojo
                team_counts[team_id].add(tid)
                labels.append(f"T{team_id + 1} #{tid}")
            elif cls == "goalkeeper":
                new_class_ids[j] = 2                # amarillo
                labels.append(f"GK #{tid}")
            elif cls == "referee":
                new_class_ids[j] = 2                # amarillo
                labels.append(f"REF #{tid}")
            else:                                   # ball
                new_class_ids[j] = 0
                labels.append(f"#{tid}")

        detections.class_id = new_class_ids

        ball_mask    = np.array([c == "ball" for c in classes])
        non_ball_det = detections[~ball_mask]
        ball_det     = detections[ball_mask]
        non_ball_labels = [l for l, m in zip(labels, (~ball_mask).tolist()) if m]

        annotated = frame.copy()
        if len(non_ball_det) > 0:
            annotated = ellipse_ann.annotate(scene=annotated, detections=non_ball_det)
            annotated = label_ann.annotate(
                scene=annotated, detections=non_ball_det, labels=non_ball_labels
            )
        if len(ball_det) > 0:
            annotated = triangle_ann.annotate(scene=annotated, detections=ball_det)

        writer.write(annotated)

    writer.release()

    print(f"\n{'='*45}")
    print(f"Frames procesados     : {frame_idx}")
    print(f"Balón detectado       : {ball_detected}/{frame_idx} frames")
    print(f"Balón interpolado     : {ball_filled} frames adicionales")
    print(f"Jugadores Equipo 1    : {len(team_counts[0])} IDs únicos")
    print(f"Jugadores Equipo 2    : {len(team_counts[1])} IDs únicos")
    print(f"IDs únicos cacheados  : {len(classifier._cache)}")
    print(f"Vídeo guardado en     : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      type=Path,  default=INPUT_VIDEO)
    parser.add_argument("--output",     type=Path,  default=OUTPUT_VIDEO)
    parser.add_argument("--conf",       type=float, default=CONF_THRESH)
    parser.add_argument("--max-frames", type=int,   default=0)
    parser.add_argument("--start-sec",  type=int,   default=0)
    args = parser.parse_args()
    run(args.input, args.output, args.conf, args.max_frames, args.start_sec)
