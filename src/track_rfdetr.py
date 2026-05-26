"""
Detección + tracking con RF-DETR SoccerNet (julianzu9612/RFDETR-Soccernet).

Pipeline idéntico a track.py excepto en el backend de inferencia:
  - Modelo: RF-DETR-Large fine-tuned en SoccerNet-Tracking-2023 (mAP@50 85.7%)
  - Carga: checkpoint descargado de HuggingFace Hub (HF_TOKEN en .env)
  - Salida: sv.Detections con class_name inyectado desde class_id

Mantiene intactos:
- ByteTrack (mismos parámetros broadcast)
- Detección de cortes de cámara + reset
- TeamClassifier: SigLIP → UMAP → KMeans
- Interpolación de balón, posesión, anotación visual

Uso: python src/track_rfdetr.py [--input PATH] [--output PATH] [--conf FLOAT]
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
import torch
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from team_assigner import TeamClassifier
from team_assigner_kmeans import TeamClassifierKMeans

load_dotenv()

# ── Modelo RF-DETR ────────────────────────────────────────────────────────────
RFDETR_HF_REPO = "julianzu9612/RFDETR-Soccernet"
RFDETR_CKPT    = "weights/checkpoint_best_regular.pth"
RFDETR_CLASSES = ["ball", "player", "referee", "goalkeeper"]

# ── Configuración general ─────────────────────────────────────────────────────
INPUT_VIDEO  = Path("data/videos/tactico_01.mp4")
OUTPUT_DIR   = Path("outputs/tracked_videos")
OUTPUT_VIDEO = OUTPUT_DIR / "tactico_01_rfdetr.mp4"
CONF_THRESH           = 0.35
CONF_THRESH_BALL      = 0.05   # RF-DETR da confidencias bajas en balón (~0.05-0.13)
POSSESSION_PROXIMITY  = 30

# ── Detección de cortes de cámara ─────────────────────────────────────────────
RESET_ON_CUT  = True
CUT_THRESHOLD = 30.0

# ── Parámetros ByteTrack ──────────────────────────────────────────────────────
BT_ACTIVATION_THRESH = 0.25
BT_LOST_BUFFER       = 60
BT_MATCH_THRESH      = 0.8

# ── Re-ID por apariencia ──────────────────────────────────────────────────────
REID_COSINE_THRESH = 0.85

# ── Paleta de colores ─────────────────────────────────────────────────────────
COLOR_PALETTE = sv.ColorPalette.from_hex(["#00FF00", "#FF4444", "#FFFF00"])
BALL_COLOR    = sv.Color.from_hex("#FFFFFF")

_ERA_OFFSET = 100_000


# ── Carga del modelo RF-DETR ──────────────────────────────────────────────────

def _load_rfdetr(device: str):
    """
    Descarga el checkpoint de HuggingFace y carga RF-DETR con 4 clases SoccerNet.
    Sigue el patrón oficial de julianzu9612/RFDETR-Soccernet/inference.py:
      RFDETRBase() → reinitialize_detection_head(4) → cargar checkpoint
    """
    from rfdetr import RFDETRLargeDeprecated
    from huggingface_hub import hf_hub_download

    token = os.getenv("HF_TOKEN")
    print(f"  [RFDETR] Descargando checkpoint desde {RFDETR_HF_REPO}...")
    ckpt_path = hf_hub_download(RFDETR_HF_REPO, RFDETR_CKPT, token=token)
    print(f"  [RFDETR] Checkpoint en: {ckpt_path}")

    print(f"  [RFDETR] Inicializando modelo en {device}...")
    model = RFDETRLargeDeprecated(pretrain_weights=None)
    model.model.model.reinitialize_detection_head(len(RFDETR_CLASSES))

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.model.model.load_state_dict(state)
    model.model.model.to(device)
    model.model.model.eval()
    print("  [RFDETR] Modelo listo.")
    return model


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_rfdetr(model, frame: np.ndarray) -> sv.Detections:
    """
    Inferencia RF-DETR. Convierte BGR→RGB, llama a predict() y añade
    class_name a detections.data para que el resto del pipeline funcione igual.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = model.predict(frame_rgb, threshold=0.01)
    if len(detections) == 0:
        return detections
    detections.data["class_name"] = np.array(
        [RFDETR_CLASSES[int(cid)] for cid in detections.class_id]
    )
    return detections


def interpolate_ball(ball_bboxes: list) -> list:
    df = pd.DataFrame(
        [b if b is not None else [np.nan] * 4 for b in ball_bboxes],
        columns=["x1", "y1", "x2", "y2"],
    )
    df = df.interpolate().bfill()
    result = []
    for _, row in df.iterrows():
        result.append(None if np.isnan(row["x1"]) else row.tolist())
    return result


def _apply_conf_filter(detections: sv.Detections, conf: float) -> sv.Detections:
    classes   = detections.data.get("class_name", np.array([]))
    ball_mask = np.array([c == "ball" for c in classes], dtype=bool)
    non_ball_keep = ~ball_mask & (detections.confidence >= conf)

    # Solo la detección de balón con mayor confianza (evita falsos positivos múltiples)
    ball_keep = np.zeros(len(detections), dtype=bool)
    ball_indices = np.where(ball_mask & (detections.confidence >= CONF_THRESH_BALL))[0]
    if len(ball_indices) > 0:
        best = ball_indices[np.argmax(detections.confidence[ball_indices])]
        ball_keep[best] = True

    return detections[non_ball_keep | ball_keep]


def _make_tracker() -> sv.ByteTrack:
    return sv.ByteTrack(
        track_activation_threshold=BT_ACTIVATION_THRESH,
        lost_track_buffer=BT_LOST_BUFFER,
        minimum_matching_threshold=BT_MATCH_THRESH,
    )


def detect_cut(prev_frame: np.ndarray, curr_frame: np.ndarray,
               threshold: float = CUT_THRESHOLD) -> bool:
    return float(cv2.absdiff(prev_frame, curr_frame).mean()) > threshold


def reassign_ids_by_appearance(
    new_detections: sv.Detections,
    frame: np.ndarray,
    snapshot_embeddings: dict[int, np.ndarray],
    classifier: TeamClassifier,
    threshold: float = REID_COSINE_THRESH,
    frame_idx: int = 0,
) -> tuple[dict[int, int], list[float]]:
    if not snapshot_embeddings or new_detections.tracker_id is None:
        return {}, []
    classes    = new_detections.data.get("class_name", np.array([]))
    player_idx = [j for j, c in enumerate(classes) if c == "player"]
    if not player_idx:
        return {}, []
    crops    = [sv.crop_image(frame, new_detections.xyxy[j]) for j in player_idx]
    new_embs = classifier._extract_features(crops)
    new_tids = [int(new_detections.tracker_id[j]) for j in player_idx]
    old_ids  = list(snapshot_embeddings.keys())
    old_embs = np.array([snapshot_embeddings[oid] for oid in old_ids])
    old_norm = old_embs / (np.linalg.norm(old_embs, axis=1, keepdims=True) + 1e-8)
    new_norm = new_embs / (np.linalg.norm(new_embs, axis=1, keepdims=True) + 1e-8)
    sim      = new_norm @ old_norm.T
    all_sims = sim.ravel().tolist()
    remap: dict[int, int] = {}
    used_old: set[int]    = set()
    for flat_idx in np.argsort(sim.ravel())[::-1]:
        i_new, i_old = divmod(int(flat_idx), len(old_ids))
        if sim[i_new, i_old] < threshold:
            break
        new_tid, old_tid = new_tids[i_new], old_ids[i_old]
        if new_tid not in remap and old_tid not in used_old:
            remap[new_tid] = old_tid
            used_old.add(old_tid)
            classifier._emb_cache[old_tid] = new_embs[i_new]
    return remap, all_sims


# ── Pipeline principal ────────────────────────────────────────────────────────

def run(input_path: Path, output_path: Path, conf: float,
        max_frames: int = 0, start_sec: int = 0, assigner: str = "siglip",
        no_video: bool = False) -> None:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = _load_rfdetr(device)
    infer  = lambda frame: _infer_rfdetr(model, frame)

    tracker    = _make_tracker()
    if assigner == "kmeans":
        classifier = TeamClassifierKMeans()
        print(f"Clasificador de equipo: KMeans (píxeles)")
    else:
        classifier = TeamClassifier()
        print(f"Clasificador de equipo: SigLIP+UMAP+KMeans")

    cap   = cv2.VideoCapture(str(input_path))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_sec:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))

    limit = max_frames if max_frames else (total - int(start_sec * fps))
    pasadas = "1/1 (solo CSV)" if no_video else "1/2"
    print(f"Pasada {pasadas} — inferencia ({limit} frames desde seg {start_sec})...")
    if no_video:
        print("  Modo --no-video: sin renderizado, máxima velocidad.")

    frames_buf: list[np.ndarray]        = []  # vacío si no_video=True
    detections_buf: list[sv.Detections] = []
    team_buf: list[dict[int, int]]      = []
    disp_buf: list[dict[int, int]]      = []
    ball_bboxes: list                   = []
    fitted            = False
    fit_crops_accum:  list[np.ndarray] = []
    fit_frames_seen   = 0
    FIT_FRAMES_NEEDED = 5
    FIT_MIN_PLAYERS   = 4

    prev_frame: np.ndarray | None              = None
    era                                        = 0
    global_remap: dict[int, int]               = {}
    snapshot_embeddings: dict[int, np.ndarray] = {}
    last_player_gtids: list[int]               = []
    pending_reid                               = False
    n_cuts = n_reids                           = 0
    all_sims_observed: list[float]             = []

    frame_idx = 0
    while frame_idx < limit:
        ret, frame = cap.read()
        if not ret:
            break

        if RESET_ON_CUT and prev_frame is not None and detect_cut(prev_frame, frame):
            n_cuts += 1
            print(f"  [CUT] Corte en frame {frame_idx} (total: {n_cuts})")
            snapshot_embeddings = {
                eff: classifier._emb_cache[eff]
                for gtid in last_player_gtids
                for eff in (global_remap.get(gtid, gtid),)
                if eff in classifier._emb_cache
            }
            era += 1
            tracker = _make_tracker()
            pending_reid = True

        detections = infer(frame)
        detections = _apply_conf_filter(detections, conf)
        detections = tracker.update_with_detections(detections)

        classes = detections.data.get("class_name", np.array([]))

        if pending_reid:
            if classifier._fitted and snapshot_embeddings and hasattr(classifier, "_extract_features"):
                cut_remap, cut_sims = reassign_ids_by_appearance(
                    detections, frame, snapshot_embeddings, classifier,
                    frame_idx=frame_idx,
                )
                all_sims_observed.extend(cut_sims)
                for raw_new, old_eff in cut_remap.items():
                    global_remap[raw_new + era * _ERA_OFFSET] = old_eff
                n_reids += len(cut_remap)
                if cut_remap:
                    print(f"  [ReID] {len(cut_remap)} IDs reasignados en frame {frame_idx}")
            pending_reid = False

        if not fitted and fit_frames_seen < FIT_FRAMES_NEEDED:
            player_mask = np.array([c == "player" for c in classes], dtype=bool)
            if player_mask.sum() >= FIT_MIN_PLAYERS:
                fit_crops_accum += [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy[player_mask]]
                fit_frames_seen += 1
                if fit_frames_seen >= FIT_FRAMES_NEEDED:
                    classifier.fit(fit_crops_accum)
                    fitted = True

        frame_teams:   dict[int, int] = {}
        frame_display: dict[int, int] = {}
        frame_counts:  dict[int, int] = {0: 0, 1: 0}
        curr_player_gtids: list[int]  = []

        if detections.tracker_id is not None:
            for j, (cls, tid) in enumerate(zip(classes, detections.tracker_id)):
                raw_tid = int(tid)
                gtid    = raw_tid + era * _ERA_OFFSET
                eff_tid = global_remap.get(gtid, gtid)
                disp_id = eff_tid % _ERA_OFFSET
                if cls == "player":
                    crop    = sv.crop_image(frame, detections.xyxy[j])
                    team_id = classifier.get_team_with_limit(crop, eff_tid, frame_counts)
                    frame_teams[raw_tid]  = team_id
                    frame_counts[team_id] = frame_counts.get(team_id, 0) + 1
                    curr_player_gtids.append(gtid)
                frame_display[raw_tid] = disp_id

        last_player_gtids = curr_player_gtids

        ball_bbox = None
        for j, cls in enumerate(classes):
            if cls == "ball":
                ball_bbox = detections.xyxy[j].tolist()
                break
        ball_bboxes.append(ball_bbox)

        if not no_video:
            frames_buf.append(frame)   # solo buffear frame si hay que renderizar
        detections_buf.append(detections)
        team_buf.append(frame_teams)
        disp_buf.append(frame_display)
        prev_frame = frame
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{limit} frames | IDs: {len(classifier._cache)} | Cortes: {n_cuts}")

    cap.release()

    print("Interpolando posición del balón...")
    ball_interp   = interpolate_ball(ball_bboxes)
    ball_detected = sum(1 for b in ball_bboxes if b is not None)
    ball_filled   = sum(1 for b in ball_interp if b is not None) - ball_detected

    # ── Modo CSV-only: generar CSV sin renderizar vídeo ──────────────────────
    if no_video:
        csv_rows: list[dict] = []
        ball_detected = sum(1 for b in ball_bboxes if b is not None)
        for i, (detections, frame_teams, frame_display) in enumerate(
            zip(detections_buf, team_buf, disp_buf)
        ):
            classes = detections.data.get("class_name", np.array([]))
            for j, cls in enumerate(classes):
                raw_tid = int(detections.tracker_id[j]) if detections.tracker_id is not None else -1
                disp_id = frame_display.get(raw_tid, raw_tid)
                team_id = frame_teams.get(raw_tid, -1) if cls == "player" else -1
                x1, y1, x2, y2 = detections.xyxy[j]
                conf_val = float(detections.confidence[j]) if detections.confidence is not None else 0.0
                csv_rows.append({
                    "frame": i, "track_id": disp_id, "class": cls, "team_id": team_id,
                    "x1": round(float(x1), 1), "y1": round(float(y1), 1),
                    "x2": round(float(x2), 1), "y2": round(float(y2), 1),
                    "confidence": round(conf_val, 4),
                })
            if (i + 1) % 5000 == 0:
                print(f"  CSV: {i+1}/{len(detections_buf)} frames escritos...")
        csv_path = Path("outputs/metrics") / (output_path.stem + "_tracks.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"\nCSV guardado en: {csv_path}  ({len(csv_rows)} filas)")
        print(f"Frames procesados : {frame_idx}  |  Balón detectado: {ball_detected} frames")
        return

    print(f"Pasada 2/2 — renderizando → {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    ellipse_ann    = sv.EllipseAnnotator(color=COLOR_PALETTE, thickness=2)
    label_ann      = sv.LabelAnnotator(
        color=COLOR_PALETTE, text_color=sv.Color.BLACK,
        text_scale=0.45, text_thickness=1, text_padding=3,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    triangle_ann   = sv.TriangleAnnotator(color=BALL_COLOR, base=20, height=17, outline_thickness=1)
    possession_ann = sv.TriangleAnnotator(
        color=sv.Color.from_hex("#FFFF00"), base=25, height=21,
        position=sv.Position.TOP_CENTER, outline_thickness=2,
    )

    team_counts: dict[int, set] = defaultdict(set)
    csv_rows: list[dict] = []

    for i, (frame, detections) in enumerate(zip(frames_buf, detections_buf)):
        ball_bbox = ball_interp[i]
        if ball_bbox is not None and "class_name" in detections.data:
            for j, cls in enumerate(detections.data["class_name"]):
                if cls == "ball":
                    detections.xyxy[j] = np.array(ball_bbox)
                    break

        classes       = detections.data.get("class_name", np.array([]))
        frame_teams   = team_buf[i]
        frame_display = disp_buf[i]
        n             = len(detections)
        new_class_ids = np.zeros(n, dtype=int)
        labels        = []

        for j, cls in enumerate(classes):
            raw_tid = int(detections.tracker_id[j]) if detections.tracker_id is not None else -1
            disp_id = frame_display.get(raw_tid, raw_tid)
            if cls == "player":
                team_id = frame_teams.get(raw_tid, 0)
                new_class_ids[j] = team_id
                team_counts[team_id].add(disp_id)
                labels.append(f"T{team_id + 1} #{disp_id}")
            elif cls == "goalkeeper":
                new_class_ids[j] = 2
                labels.append(f"GK #{disp_id}")
            elif cls == "referee":
                new_class_ids[j] = 2
                labels.append(f"REF #{disp_id}")
            else:
                new_class_ids[j] = 0
                labels.append(f"#{disp_id}")

        detections.class_id = new_class_ids

        ball_mask       = np.array([c == "ball" for c in classes], dtype=bool)
        non_ball_det    = detections[~ball_mask]
        ball_det        = detections[ball_mask]
        non_ball_labels = [l for l, m in zip(labels, (~ball_mask).tolist()) if m]

        possession_det = sv.Detections.empty()
        if ball_bbox is not None:
            bx1, by1, bx2, by2 = ball_bbox
            ball_cx, ball_cy = (bx1 + bx2) / 2, (by1 + by2) / 2
            best_j, best_dist = None, float("inf")
            P = POSSESSION_PROXIMITY
            for j, cls in enumerate(classes):
                if cls in ("player", "goalkeeper") and detections.tracker_id is not None:
                    x1, y1, x2, y2 = detections.xyxy[j]
                    if x1 - P < ball_cx < x2 + P and y1 - P < ball_cy < y2 + P:
                        dist = ((x1 + x2) / 2 - ball_cx) ** 2 + ((y1 + y2) / 2 - ball_cy) ** 2
                        if dist < best_dist:
                            best_dist, best_j = dist, j
            if best_j is not None:
                possession_det = detections[[best_j]]

        # Acumular filas CSV
        for j, cls in enumerate(classes):
            raw_tid = int(detections.tracker_id[j]) if detections.tracker_id is not None else -1
            disp_id = frame_display.get(raw_tid, raw_tid)
            team_id = frame_teams.get(raw_tid, -1) if cls == "player" else -1
            x1, y1, x2, y2 = detections.xyxy[j]
            conf = float(detections.confidence[j]) if detections.confidence is not None else 0.0
            csv_rows.append({
                "frame": i, "track_id": disp_id, "class": cls, "team_id": team_id,
                "x1": round(float(x1), 1), "y1": round(float(y1), 1),
                "x2": round(float(x2), 1), "y2": round(float(y2), 1),
                "confidence": round(conf, 4),
            })

        annotated = frame.copy()
        if len(non_ball_det) > 0:
            annotated = ellipse_ann.annotate(scene=annotated, detections=non_ball_det)
            annotated = label_ann.annotate(scene=annotated, detections=non_ball_det, labels=non_ball_labels)
        if len(ball_det) > 0:
            annotated = triangle_ann.annotate(scene=annotated, detections=ball_det)
        if len(possession_det) > 0:
            annotated = possession_ann.annotate(scene=annotated, detections=possession_det)

        writer.write(annotated)

    writer.release()

    # Guardar CSV
    csv_path = Path("outputs/metrics") / (output_path.stem + "_tracks.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"CSV de tracks guardado en: {csv_path}")

    total_ids = len(team_counts[0]) + len(team_counts[1])
    print(f"\n{'='*50}")
    print(f"[MODEL] RFDETR-Soccernet | frames procesados: {frame_idx} | IDs únicos: {total_ids} | balón detectado: {ball_detected} frames")
    print(f"{'='*50}")
    print(f"Frames procesados     : {frame_idx}")
    print(f"Balón detectado       : {ball_detected}/{frame_idx} frames")
    print(f"Balón interpolado     : {ball_filled} frames adicionales")
    print(f"Jugadores Equipo 1    : {len(team_counts[0])} IDs únicos")
    print(f"Jugadores Equipo 2    : {len(team_counts[1])} IDs únicos")
    print(f"IDs únicos cacheados  : {len(classifier._cache)}")
    print(f"Cortes de cámara      : {n_cuts}")
    print(f"IDs reasignados Re-ID : {n_reids}")
    if all_sims_observed:
        print(f"[REID SUMMARY] similitudes: min={min(all_sims_observed):.3f}, max={max(all_sims_observed):.3f}, media={np.mean(all_sims_observed):.3f}")
    print(f"Vídeo guardado en     : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      type=Path,  default=INPUT_VIDEO)
    parser.add_argument("--output",     type=Path,  default=OUTPUT_VIDEO)
    parser.add_argument("--conf",       type=float, default=CONF_THRESH)
    parser.add_argument("--max-frames", type=int,   default=0)
    parser.add_argument("--start-sec",  type=int,   default=0)
    parser.add_argument("--no-video",   action="store_true",
                        help="Solo genera el CSV sin renderizar vídeo (mucho más rápido y sin coste de RAM de frames)")
    parser.add_argument("--assigner",   type=str,   default="siglip",
                        choices=["siglip", "kmeans"],
                        help="Clasificador de equipos: siglip (SigLIP+UMAP+KMeans) o kmeans (píxeles)")
    args = parser.parse_args()
    run(args.input, args.output, args.conf, args.max_frames, args.start_sec, args.assigner,
        no_video=args.no_video)
