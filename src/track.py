"""
Detección + tracking sobre vídeo táctico con ByteTrack (supervision).

Modos de inferencia (controlar con USE_LOCAL_MODEL):
  True  → ultralytics YOLO local (models/football-tactical-v4.pt) — sin coste de API
  False → Roboflow Inference API (football-tactical/4)             — requiere créditos

Incluye:
- Interpolación de posición del balón (abdullahtarek/football-analysis)
- Clasificación de equipo: SigLIP → UMAP → KMeans, cacheado por track_id
- Threshold diferenciado: CONF_THRESH jugadores/árbitros, CONF_THRESH_BALL balón
- Detección de cortes de cámara + reset limpio del tracker (RESET_ON_CUT)
- Re-ID por apariencia tras corte usando SigLIP (similitud coseno ≥ REID_COSINE_THRESH)

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
import torch
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from team_assigner import TeamClassifier
from team_assigner_kmeans import TeamClassifierKMeans

load_dotenv()

# ── Alternar entre inferencia local y API ────────────────────────────────────
USE_LOCAL_MODEL = True

# Local
LOCAL_MODEL_PATH = Path("models/football-tactical-v4.pt")

# API (solo si USE_LOCAL_MODEL = False)
API_KEY  = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = "football-tactical/4"
API_URL  = "https://serverless.roboflow.com"

# ── Configuración general ─────────────────────────────────────────────────────
INPUT_VIDEO  = Path("data/videos/tactico_01.mp4")
OUTPUT_DIR   = Path("outputs/tracked_videos")
OUTPUT_VIDEO = OUTPUT_DIR / "tactico_01_tracked.mp4"
CONF_THRESH           = 0.35   # jugadores, árbitros, porteros
CONF_THRESH_BALL      = 0.05   # balón: umbral muy agresivo para no perdérselo
POSSESSION_PROXIMITY  = 30     # px — bounding box padded para decidir posesión del balón

# ── Detección de cortes de cámara ─────────────────────────────────────────────
RESET_ON_CUT  = True
CUT_THRESHOLD = 30.0      # diff.mean() a partir del cual se considera corte de cámara

# ── Parámetros ByteTrack ──────────────────────────────────────────────────────
BT_ACTIVATION_THRESH = 0.25
BT_LOST_BUFFER       = 90
BT_MATCH_THRESH      = 0.8

# ── Re-ID por apariencia ──────────────────────────────────────────────────────
REID_COSINE_THRESH = 0.85

# ── Paleta de colores ─────────────────────────────────────────────────────────
COLOR_PALETTE = sv.ColorPalette.from_hex(["#00FF00", "#FF4444", "#FFFF00"])
BALL_COLOR    = sv.Color.from_hex("#FFFFFF")

# Offset para garantizar unicidad global de track_ids entre eras (resets del tracker).
# ByteTrack reinicia su contador a 1 en cada nueva instancia; el offset evita colisiones
# de caché entre el jugador #3 del tracker anterior y el nuevo jugador #3.
_ERA_OFFSET = 100_000


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _infer_local(model, frame: np.ndarray) -> sv.Detections:
    """Inferencia con ultralytics YOLO local. Retorna todas las detecciones (conf ≥ 0)."""
    results = model.predict(frame, conf=0.01, verbose=False)[0]
    return sv.Detections.from_ultralytics(results)


def _infer_api(client, frame: np.ndarray) -> sv.Detections:
    """Inferencia vía Roboflow Inference API."""
    results = client.infer(frame, model_id=MODEL_ID)
    return sv.Detections.from_inference(results)


def _apply_conf_filter(detections: sv.Detections, conf: float) -> sv.Detections:
    """Threshold diferenciado: CONF_THRESH_BALL para balón, conf para el resto."""
    classes   = detections.data.get("class_name", np.array([]))
    ball_mask = np.array([c == "ball" for c in classes], dtype=bool)
    keep      = (
        (~ball_mask & (detections.confidence >= conf)) |
        ( ball_mask & (detections.confidence >= CONF_THRESH_BALL))
    )
    return detections[keep]


def _make_tracker() -> sv.ByteTrack:
    return sv.ByteTrack(
        track_activation_threshold=BT_ACTIVATION_THRESH,
        lost_track_buffer=BT_LOST_BUFFER,
        minimum_matching_threshold=BT_MATCH_THRESH,
    )


def detect_cut(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    threshold: float = CUT_THRESHOLD,
) -> bool:
    """True si la diferencia media entre frames supera el umbral."""
    diff = cv2.absdiff(prev_frame, curr_frame)
    return float(diff.mean()) > threshold


def reassign_ids_by_appearance(
    new_detections: sv.Detections,
    frame: np.ndarray,
    snapshot_embeddings: dict[int, np.ndarray],
    classifier: TeamClassifier,
    threshold: float = REID_COSINE_THRESH,
    frame_idx: int = 0,
) -> tuple[dict[int, int], list[float]]:
    """
    Greedy matching por similitud coseno entre jugadores post-corte y snapshot.

    Extrae embeddings SigLIP de los jugadores del nuevo frame y los compara contra
    snapshot_embeddings (effective_tid → embedding_768D del frame previo al corte).
    Asignación greedy en orden descendente de similitud; cada ID se usa como máximo una vez.

    Args:
        new_detections:      detecciones del primer frame tras corte (raw track_ids nuevos)
        snapshot_embeddings: {effective_tid: embedding_768D} guardado antes del corte
        threshold:           similitud coseno mínima para aceptar un match
    Returns:
        ({raw_new_tid: old_effective_tid}, [todas las similitudes observadas])
    """
    if not snapshot_embeddings or new_detections.tracker_id is None:
        return {}, []

    classes    = new_detections.data.get("class_name", np.array([]))
    player_idx = [j for j, c in enumerate(classes) if c == "player"]
    if not player_idx:
        return {}, []

    crops    = [sv.crop_image(frame, new_detections.xyxy[j]) for j in player_idx]
    new_embs = classifier._extract_features(crops)                    # (N_new, 768)
    new_tids = [int(new_detections.tracker_id[j]) for j in player_idx]

    old_ids  = list(snapshot_embeddings.keys())
    old_embs = np.array([snapshot_embeddings[oid] for oid in old_ids])  # (N_old, 768)

    old_norm = old_embs / (np.linalg.norm(old_embs, axis=1, keepdims=True) + 1e-8)
    new_norm = new_embs / (np.linalg.norm(new_embs, axis=1, keepdims=True) + 1e-8)
    sim      = new_norm @ old_norm.T                                   # (N_new, N_old)

    all_sims = sim.ravel().tolist()

    # ── Greedy matching ───────────────────────────────────────────────────────
    remap: dict[int, int] = {}
    used_old: set[int]    = set()
    for flat_idx in np.argsort(sim.ravel())[::-1]:
        i_new, i_old = divmod(int(flat_idx), len(old_ids))
        if sim[i_new, i_old] < threshold:
            break
        new_tid = new_tids[i_new]
        old_tid = old_ids[i_old]
        if new_tid not in remap and old_tid not in used_old:
            remap[new_tid] = old_tid
            used_old.add(old_tid)
            # Actualiza embedding del ID efectivo con la nueva observación
            classifier._emb_cache[old_tid] = new_embs[i_new]

    return remap, all_sims


# ── Pipeline principal ────────────────────────────────────────────────────────

def run(input_path: Path, output_path: Path, conf: float,
        max_frames: int = 0, start_sec: int = 0, assigner: str = "siglip") -> None:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Inicializar backend de inferencia
    if USE_LOCAL_MODEL:
        from ultralytics import YOLO
        model  = YOLO(str(LOCAL_MODEL_PATH))
        model.to(device)
        infer  = lambda frame: _infer_local(model, frame)
        print(f"Modo: LOCAL — {LOCAL_MODEL_PATH.name} en {device}")
    else:
        from inference_sdk import InferenceHTTPClient
        client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)
        infer  = lambda frame: _infer_api(client, frame)
        print(f"Modo: API   — {MODEL_ID}")

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
    print(f"Pasada 1/2 — inferencia ({limit} frames desde seg {start_sec})...")

    frames_buf: list[np.ndarray]        = []
    detections_buf: list[sv.Detections] = []
    team_buf: list[dict[int, int]]      = []   # raw_tid → team_id
    disp_buf: list[dict[int, int]]      = []   # raw_tid → display_id
    ball_bboxes: list                   = []
    fitted            = False
    fit_crops_accum:  list[np.ndarray] = []
    fit_frames_seen   = 0
    FIT_FRAMES_NEEDED = 5
    FIT_MIN_PLAYERS   = 4

    # ── Estado cortes / Re-ID ─────────────────────────────────────────────────
    prev_frame: np.ndarray | None              = None
    era                                        = 0     # se incrementa en cada corte
    global_remap: dict[int, int]               = {}    # global_tid → effective_tid
    snapshot_embeddings: dict[int, np.ndarray] = {}    # effective_tid → emb_768D
    last_player_gtids: list[int]               = []    # global_tids visibles en frame anterior
    pending_reid                               = False
    n_cuts                                     = 0
    n_reids                                    = 0
    all_sims_observed: list[float]             = []    # para el resumen de calibración

    frame_idx = 0
    while frame_idx < limit:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Detección de corte ────────────────────────────────────────────────
        if RESET_ON_CUT and prev_frame is not None and detect_cut(prev_frame, frame):
            n_cuts += 1
            print(f"  [CUT] Corte de cámara en frame {frame_idx} (total: {n_cuts})")
            snapshot_embeddings = {
                eff: classifier._emb_cache[eff]
                for gtid in last_player_gtids
                for eff in (global_remap.get(gtid, gtid),)
                if eff in classifier._emb_cache
            }
            era     += 1
            tracker  = _make_tracker()
            pending_reid = True

        detections = infer(frame)
        detections = _apply_conf_filter(detections, conf)
        detections = tracker.update_with_detections(detections)

        classes = detections.data.get("class_name", np.array([]))

        # ── Re-ID por apariencia (solo SigLIP; KMeans no tiene _extract_features) ──
        if pending_reid:
            if classifier._fitted and snapshot_embeddings and hasattr(classifier, "_extract_features"):
                cut_remap, cut_sims = reassign_ids_by_appearance(
                    detections, frame, snapshot_embeddings, classifier,
                    frame_idx=frame_idx,
                )
                all_sims_observed.extend(cut_sims)
                for raw_new, old_eff in cut_remap.items():
                    gtid_new = raw_new + era * _ERA_OFFSET
                    global_remap[gtid_new] = old_eff
                n_reids += len(cut_remap)
                if cut_remap:
                    print(f"  [ReID] {len(cut_remap)} IDs reasignados en frame {frame_idx}")
            pending_reid = False

        # ── Fit del clasificador: acumula 5 frames con ≥4 jugadores ─────────
        if not fitted and fit_frames_seen < FIT_FRAMES_NEEDED:
            class_counts: dict[str, int] = {}
            for c in classes:
                class_counts[c] = class_counts.get(c, 0) + 1
            player_mask = np.array([c == "player" for c in classes], dtype=bool)
            n_players = int(player_mask.sum())
            print(f"  [FIT DEBUG f{frame_idx}] Detecciones por clase: {class_counts} → players válidos: {n_players}")

            if n_players >= FIT_MIN_PLAYERS:
                frame_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy[player_mask]]

                # Validación anti-verde: si algún cluster del lote tiene G > R y G > B, descarta
                import cv2 as _cv2
                batch_colors = []
                for crop in frame_crops:
                    top = crop[: crop.shape[0] // 2, :]
                    mean_bgr = top.reshape(-1, 3).mean(axis=0)
                    batch_colors.append(mean_bgr)
                batch_colors = np.array(batch_colors)
                mean_b, mean_g, mean_r = batch_colors.mean(axis=0)
                if mean_g > mean_r and mean_g > mean_b:
                    print(f"  [FIT DEBUG f{frame_idx}] Lote descartado — verde dominante "
                          f"(R={mean_r:.0f} G={mean_g:.0f} B={mean_b:.0f})")
                else:
                    fit_crops_accum += frame_crops
                    fit_frames_seen += 1
                    print(f"  [FIT DEBUG f{frame_idx}] Lote aceptado ({n_players} crops) "
                          f"— frame {fit_frames_seen}/{FIT_FRAMES_NEEDED}")
                    if fit_frames_seen >= FIT_FRAMES_NEEDED:
                        classifier.fit(fit_crops_accum)
                        fitted = True

        # ── Asignación de equipo con restricción de cardinalidad ──────────────
        frame_teams:   dict[int, int] = {}   # raw_tid → team_id
        frame_display: dict[int, int] = {}   # raw_tid → display_id
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
                    frame_teams[raw_tid]   = team_id
                    frame_counts[team_id]  = frame_counts.get(team_id, 0) + 1
                    curr_player_gtids.append(gtid)
                frame_display[raw_tid] = disp_id

        last_player_gtids = curr_player_gtids

        # ── Balón para interpolación ──────────────────────────────────────────
        ball_bbox = None
        for j, cls in enumerate(classes):
            if cls == "ball":
                ball_bbox = detections.xyxy[j].tolist()
                break
        ball_bboxes.append(ball_bbox)

        frames_buf.append(frame)
        detections_buf.append(detections)
        team_buf.append(frame_teams)
        disp_buf.append(frame_display)

        prev_frame = frame
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(
                f"  {frame_idx}/{limit} frames"
                f" | IDs cacheados: {len(classifier._cache)}"
                f" | Cortes: {n_cuts}"
            )

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

    ellipse_ann     = sv.EllipseAnnotator(color=COLOR_PALETTE, thickness=2)
    label_ann       = sv.LabelAnnotator(
        color=COLOR_PALETTE,
        text_color=sv.Color.BLACK,
        text_scale=0.45,
        text_thickness=1,
        text_padding=3,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    triangle_ann    = sv.TriangleAnnotator(
        color=BALL_COLOR, base=20, height=17, outline_thickness=1
    )
    # Triángulo sobre el jugador en posesión del balón (idea: roboflow/notebooks football tracker)
    possession_ann  = sv.TriangleAnnotator(
        color=sv.Color.from_hex("#FFFF00"), base=25, height=21,
        position=sv.Position.TOP_CENTER, outline_thickness=2,
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

        # ── Posesión del balón ────────────────────────────────────────────────
        # Jugador cuyo bounding box (con margen POSSESSION_PROXIMITY) contiene
        # el centro del balón — misma lógica que roboflow/notebooks football tracker
        possession_det = sv.Detections.empty()
        if ball_bbox is not None:
            bx1, by1, bx2, by2 = ball_bbox
            ball_cx = (bx1 + bx2) / 2
            ball_cy = (by1 + by2) / 2
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

        annotated = frame.copy()
        if len(non_ball_det) > 0:
            annotated = ellipse_ann.annotate(scene=annotated, detections=non_ball_det)
            annotated = label_ann.annotate(
                scene=annotated, detections=non_ball_det, labels=non_ball_labels
            )
        if len(ball_det) > 0:
            annotated = triangle_ann.annotate(scene=annotated, detections=ball_det)
        if len(possession_det) > 0:
            annotated = possession_ann.annotate(scene=annotated, detections=possession_det)

        writer.write(annotated)

    writer.release()

    print(f"\n{'='*45}")
    print(f"Modo inferencia       : {'LOCAL' if USE_LOCAL_MODEL else 'API'}")
    print(f"Frames procesados     : {frame_idx}")
    print(f"Balón detectado       : {ball_detected}/{frame_idx} frames")
    print(f"Balón interpolado     : {ball_filled} frames adicionales")
    print(f"Jugadores Equipo 1    : {len(team_counts[0])} IDs únicos")
    print(f"Jugadores Equipo 2    : {len(team_counts[1])} IDs únicos")
    print(f"IDs únicos cacheados  : {len(classifier._cache)}")
    print(f"Cortes de cámara      : {n_cuts}")
    print(f"IDs reasignados Re-ID : {n_reids}")
    if all_sims_observed:
        print(
            f"\n[REID SUMMARY] Similitudes observadas: "
            f"min={min(all_sims_observed):.3f}, "
            f"max={max(all_sims_observed):.3f}, "
            f"media={np.mean(all_sims_observed):.3f}"
        )
    print(f"Vídeo guardado en     : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      type=Path,  default=INPUT_VIDEO)
    parser.add_argument("--output",     type=Path,  default=OUTPUT_VIDEO)
    parser.add_argument("--conf",       type=float, default=CONF_THRESH)
    parser.add_argument("--max-frames", type=int,   default=0)
    parser.add_argument("--start-sec",  type=int,   default=0)
    parser.add_argument("--assigner",   type=str,   default="siglip",
                        choices=["siglip", "kmeans"],
                        help="Clasificador de equipos: siglip (SigLIP+UMAP+KMeans) o kmeans (píxeles)")
    args = parser.parse_args()
    run(args.input, args.output, args.conf, args.max_frames, args.start_sec, args.assigner)
