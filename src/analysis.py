"""
analysis.py v2 — Speed + Possession + Passes para vídeos de fútbol broadcast

Mejoras sobre v1:
  1. Posesión en ESPACIO DE PÍXELES (igual que reference repo abdullahtarek)
     → No usa homografía del balón (falla si está en el aire)
     → Distancia desde pies del jugador (esquinas inferiores del bbox) al centro del bbox del balón
     → Umbral 60px calibrado para 1280×720
  2. Interpolación lineal del balón para gaps ≤ 30f (0.5s a 59.83fps)
     → De 626 frames detectados → ~900 con posición
  3. Team ID por VOTO MAYORITARIO (corrige 22 tracks con team_id inconsistente)
  4. Velocidad con RECHAZO DE OUTLIERS por jugador
     → Descarta ventanas donde speed > 2× mediana del jugador AND speed > 25 km/h
  5. Pases: min_possession_frames=8 (era 3), min_pass_m=3.0, max_pass_m=50.0

Referencia: https://github.com/abdullahtarek/football_analysis
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))
from homography import (
    DynamicHomographyMapper,
    KP_MODEL_DYNAMIC,
    COLOR_TEAM0,
    COLOR_TEAM1,
    COLOR_DARK,
)

_PLAYER_CLASSES = {"player", "goalkeeper"}

_CFG_LEN_CM = 12000.0
_CFG_WID_CM = 7000.0
_REAL_LEN_M = 105.0
_REAL_WID_M = 68.0


def cfg_to_real_m(x_cfg: float, y_cfg: float) -> tuple[float, float]:
    return (
        x_cfg / _CFG_LEN_CM * _REAL_LEN_M,
        y_cfg / _CFG_WID_CM * _REAL_WID_M,
    )


def pos_cm_to_real_m(pos_cm: dict) -> dict:
    pos_real: dict = {}
    for frame, tids in pos_cm.items():
        pos_real[frame] = {}
        for tid, (xc, yc) in tids.items():
            pos_real[frame][tid] = cfg_to_real_m(xc, yc)
    return pos_real


def smooth_positions(pos_real_m: dict, window: int = 7) -> dict:
    """
    Media móvil centrada de ventana 'window' sobre (x_m, y_m) por track_id.

    Elimina el jitter de homografía (ruido de alta frecuencia causado por el
    paneado de cámara) sin borrar la tendencia real del movimiento del jugador.

    Algoritmo:
      - Por cada track, recoge todos los (frame, x, y) ordenados por frame.
      - Aplica rolling mean centrada: para el frame i usa el rango
        [max(0, i-half) : min(n, i+half+1)] donde half = window//2.
      - Los extremos del track se promedian con menos muestras (ventana reducida
        automáticamente), sin relleno con NaN.
    """
    # Recopilar (frame, x, y) por track
    track_data: dict[int, list] = defaultdict(list)
    for frame, tids in pos_real_m.items():
        for tid, (x, y) in tids.items():
            track_data[tid].append((frame, x, y))

    smoothed: dict = {}
    half = window // 2

    for tid, entries in track_data.items():
        entries.sort(key=lambda e: e[0])
        frames = [e[0] for e in entries]
        xs = np.array([e[1] for e in entries], dtype=np.float64)
        ys = np.array([e[2] for e in entries], dtype=np.float64)
        n = len(xs)

        xs_s = np.empty(n)
        ys_s = np.empty(n)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            xs_s[i] = xs[lo:hi].mean()
            ys_s[i] = ys[lo:hi].mean()

        for i, frame in enumerate(frames):
            if frame not in smoothed:
                smoothed[frame] = {}
            smoothed[frame][tid] = (float(xs_s[i]), float(ys_s[i]))

    return smoothed


def build_player_team(df: pd.DataFrame) -> dict[int, int]:
    """
    Asigna team_id por voto mayoritario por track_id.
    El primer team_id puede ser erróneo (SigLIP oscila en los primeros frames).
    Descarta -1 si hay otra clase disponible.
    """
    players = df[df["class"].isin(_PLAYER_CLASSES)]
    player_team: dict[int, int] = {}
    for tid, grp in players.groupby("track_id"):
        counts = grp["team_id"].value_counts()
        if len(counts) > 1 and -1 in counts.index:
            counts = counts.drop(-1)
        player_team[int(tid)] = int(counts.idxmax())
    return player_team


# ══════════════════════════════════════════════════════════════════════════════
# BallInterpolator
# ══════════════════════════════════════════════════════════════════════════════

class BallInterpolator:
    """
    Recoge la posición del balón en píxeles (centro del bbox) por frame detectado,
    e interpola linealmente los gaps de hasta max_gap frames.
    """

    def __init__(self, max_gap: int = 30):
        self.max_gap = max_gap

    def compute(self, df: pd.DataFrame, total_frames: int) -> tuple[dict, set]:
        """
        Returns:
          ball_px_per_frame : {frame: (cx, cy)} — detectados + interpolados
          detected_frames   : set de frames donde el balón fue detectado de verdad
        """
        ball_df = df[df["class"] == "ball"]
        detected: dict[int, tuple[float, float]] = {}
        for _, row in ball_df.iterrows():
            f = int(row["frame"])
            cx = (row["x1"] + row["x2"]) / 2.0
            cy = (row["y1"] + row["y2"]) / 2.0
            detected[f] = (cx, cy)

        result = dict(detected)
        frames_sorted = sorted(detected.keys())

        for i in range(len(frames_sorted) - 1):
            f0, f1 = frames_sorted[i], frames_sorted[i + 1]
            gap = f1 - f0
            if 1 < gap <= self.max_gap:
                cx0, cy0 = detected[f0]
                cx1, cy1 = detected[f1]
                for f in range(f0 + 1, f1):
                    t = (f - f0) / gap
                    result[f] = (
                        cx0 + t * (cx1 - cx0),
                        cy0 + t * (cy1 - cy0),
                    )

        print(f"  [Ball] Detectado: {len(detected)}f  "
              f"Tras interpolación (≤{self.max_gap}f): {len(result)}f")
        return result, set(detected.keys())


# ══════════════════════════════════════════════════════════════════════════════
# SpeedEstimator
# ══════════════════════════════════════════════════════════════════════════════

class SpeedEstimator:
    """
    Velocidad y distancia acumulada por jugador en metros reales.
    Ventanas NO solapadas de frame_window frames.

    v2: rechazo de outliers por jugador.
    Si speed > 2× mediana del jugador AND speed > 25 km/h → se descarta
    (jitter de homografía produce picos imposibles que este filtro elimina).
    """

    MAX_SPEED_KMH = 38.0
    MARGIN_M = 5.0

    def __init__(self, fps: float, frame_window: int = 30):
        self.fps = fps
        self.frame_window = frame_window

    def _valid(self, x_m: float, y_m: float) -> bool:
        return (
            -self.MARGIN_M < x_m < _REAL_LEN_M + self.MARGIN_M and
            -self.MARGIN_M < y_m < _REAL_WID_M + self.MARGIN_M
        )

    def compute(
        self,
        pos_real_m: dict,
        total_frames: int,
        player_ids: set,
    ) -> tuple[dict, dict]:
        # Paso 1: velocidades brutas {tid: [(w0, w1, speed_kmh, dist_m)]}
        raw: dict[int, list] = defaultdict(list)
        n_oob = n_fast = 0

        for w0 in range(0, total_frames, self.frame_window):
            w1 = min(w0 + self.frame_window, total_frames - 1)
            if w0 == w1:
                continue

            # Buscar el primer frame del segmento con homografía válida
            f0 = w0
            while f0 <= w1 and f0 not in pos_real_m:
                f0 += 1
            # Buscar el último frame del segmento con homografía válida
            f1 = w1
            while f1 >= f0 and f1 not in pos_real_m:
                f1 -= 1
            if f0 >= f1:
                continue

            common = (set(pos_real_m[f0]) & set(pos_real_m[f1])) & player_ids
            for tid in common:
                x0, y0 = pos_real_m[f0][tid]
                x1, y1 = pos_real_m[f1][tid]

                if not self._valid(x0, y0) or not self._valid(x1, y1):
                    n_oob += 1
                    continue

                dist_m = float(np.hypot(x1 - x0, y1 - y0))
                time_s = (f1 - f0) / self.fps
                speed_kmh = (dist_m / time_s) * 3.6

                if speed_kmh > self.MAX_SPEED_KMH:
                    n_fast += 1
                    continue

                raw[tid].append((w0, w1, speed_kmh, dist_m))

        # Paso 2: rechazo de outliers por jugador
        speed_lookup: dict = defaultdict(dict)
        dist_lookup: dict = defaultdict(dict)
        n_outlier = 0

        for tid, windows in raw.items():
            speeds = [s for _, _, s, _ in windows]
            median_spd = float(np.median(speeds)) if speeds else 0.0
            outlier_thresh = max(median_spd * 2.0, 25.0)

            dist_acc = 0.0
            for w0, w1, speed_kmh, dist_m in windows:
                is_outlier = (
                    len(speeds) >= 3 and
                    speed_kmh > outlier_thresh and
                    speed_kmh > 25.0
                )
                if is_outlier:
                    n_outlier += 1
                    continue
                dist_acc += dist_m
                for f in range(w0, min(w1 + 1, total_frames)):
                    speed_lookup[f][tid] = round(speed_kmh, 1)
                    dist_lookup[f][tid] = round(dist_acc, 2)

        if n_oob:
            print(f"  [Speed] {n_oob} ventanas fuera del campo descartadas")
        if n_fast:
            print(f"  [Speed] {n_fast} ventanas > {self.MAX_SPEED_KMH} km/h descartadas")
        if n_outlier:
            print(f"  [Speed] {n_outlier} outliers descartados (> 2× mediana del jugador)")

        return dict(speed_lookup), dict(dist_lookup)


# ══════════════════════════════════════════════════════════════════════════════
# PossessionTrackerPixel
# ══════════════════════════════════════════════════════════════════════════════

class PossessionTrackerPixel:
    """
    Posesión en espacio de píxeles — misma lógica que abdullahtarek/football_analysis.

    Para cada frame con posición de balón (detectada o interpolada):
      • Calcula min(dist_pie_izq, dist_pie_der) de cada jugador al centro del balón
      • Pie izq = (x1, y2),  pie der = (x2, y2)  del bbox
      • Si dist < max_dist_px → ese jugador tiene el balón

    Las estadísticas de % se calculan solo sobre frames con balón DETECTADO
    (no interpolado), para mayor fiabilidad.

    max_dist_px = 60 calibrado para 1280×720.
    El reference repo usa 70 para 1920×1080, que escala a ≈46px; usamos 60 por margen.
    """

    def __init__(self, max_dist_px: float = 60.0):
        self.max_dist_px = max_dist_px

    def compute(
        self,
        df: pd.DataFrame,
        ball_px_per_frame: dict,
        detected_frames: set,
        player_team: dict,
        total_frames: int,
    ) -> tuple[dict, dict, dict, dict]:
        # Índice de jugadores por frame: {frame: {tid: (x1,y1,x2,y2)}}
        player_rows: dict[int, dict] = defaultdict(dict)
        for row in df[df["class"].isin(_PLAYER_CLASSES)].itertuples():
            player_rows[int(row.frame)][int(row.track_id)] = (
                float(row.x1), float(row.y1), float(row.x2), float(row.y2)
            )

        raw_team: dict[int, int] = {}
        raw_player: dict[int, int] = {}

        for frame, (bx, by) in ball_px_per_frame.items():
            if frame not in player_rows:
                continue
            min_dist = float("inf")
            best_team = -1
            best_tid = -1

            for tid, (x1, y1, x2, y2) in player_rows[frame].items():
                if tid not in player_team:
                    continue
                d_left = float(np.hypot(x1 - bx, y2 - by))
                d_right = float(np.hypot(x2 - bx, y2 - by))
                d = min(d_left, d_right)
                if d < min_dist:
                    min_dist = d
                    best_tid = tid
                    best_team = player_team[tid]

            if min_dist <= self.max_dist_px:
                raw_team[frame] = best_team
                raw_player[frame] = best_tid
            else:
                raw_team[frame] = -1
                raw_player[frame] = -1

        # Estadísticas sobre frames con balón DETECTADO (no interpolado)
        counts = {0: 0, 1: 0, -1: 0}
        for f, t in raw_team.items():
            if f in detected_frames:
                counts[t] = counts.get(t, 0) + 1
        total_assigned = counts[0] + counts[1] or 1
        pct = {
            "team_0": round(counts[0] / total_assigned * 100, 1),
            "team_1": round(counts[1] / total_assigned * 100, 1),
            "none":   round(counts[-1] / (sum(counts.values()) or 1) * 100, 1),
        }

        # Sticky: propagar a todos los frames (para barra visual continua)
        possession_per_frame: dict[int, int] = {}
        player_per_frame: dict[int, int] = {}
        last_team = -1
        last_player = -1
        for f in range(total_frames):
            if f in raw_team and raw_team[f] != -1:
                last_team = raw_team[f]
                last_player = raw_player[f]
            possession_per_frame[f] = last_team
            player_per_frame[f] = last_player

        return possession_per_frame, player_per_frame, ball_px_per_frame, pct


# ══════════════════════════════════════════════════════════════════════════════
# PassTracker
# ══════════════════════════════════════════════════════════════════════════════

class PassTracker:
    """
    Detecta pases e intercepciones a partir de runs en player_per_frame.

    v2 — umbrales más estrictos:
      • min_possession_frames = 8  (antes 3) → menos falsos positivos
      • min_pass_m = 3.0            (antes 2.0)
      • max_pass_m = 50.0           (nuevo) → descarta teleportaciones por jitter
    """

    def __init__(
        self,
        min_possession_frames: int = 8,
        min_pass_m: float = 3.0,
        max_pass_m: float = 50.0,
    ):
        self.min_poss_frames = min_possession_frames
        self.min_pass_m = min_pass_m
        self.max_pass_m = max_pass_m

    def compute(
        self,
        player_per_frame: dict,
        pos_real_m: dict,
        player_team: dict,
        total_frames: int,
    ) -> tuple[list, dict]:
        # Construir runs de posesión
        runs: list[tuple[int, int, int]] = []
        current_tid = player_per_frame.get(0, -1)
        current_start = 0

        for f in range(1, total_frames):
            tid = player_per_frame.get(f, -1)
            if tid != current_tid:
                if current_tid != -1:
                    runs.append((current_tid, current_start, f - 1))
                current_tid = tid
                current_start = f

        if current_tid != -1:
            runs.append((current_tid, current_start, total_frames - 1))

        # Detectar pases entre runs consecutivos estables
        events: list[dict] = []

        for idx in range(len(runs) - 1):
            tid_a, start_a, end_a = runs[idx]
            tid_b, start_b, end_b = runs[idx + 1]

            if (end_a - start_a + 1) < self.min_poss_frames:
                continue
            if (end_b - start_b + 1) < self.min_poss_frames:
                continue
            if tid_a == tid_b or tid_a == -1 or tid_b == -1:
                continue

            # Última posición válida de A en su run
            pos_a, release_frame = None, end_a
            for f in range(end_a, start_a - 1, -1):
                if f in pos_real_m and tid_a in pos_real_m[f]:
                    pos_a = pos_real_m[f][tid_a]
                    release_frame = f
                    break

            # Primera posición válida de B en su run
            pos_b, receive_frame = None, start_b
            for f in range(start_b, end_b + 1):
                if f in pos_real_m and tid_b in pos_real_m[f]:
                    pos_b = pos_real_m[f][tid_b]
                    receive_frame = f
                    break

            if pos_a is None or pos_b is None:
                continue

            dist_m = float(np.hypot(pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]))
            if dist_m < self.min_pass_m or dist_m > self.max_pass_m:
                continue

            team_a = player_team.get(tid_a, -1)
            team_b = player_team.get(tid_b, -1)
            ev_type = (
                "pass"          if team_a == team_b and team_a != -1 else
                "interception"  if team_a != team_b and -1 not in (team_a, team_b) else
                "unknown"
            )

            ev = {
                "frame":       release_frame,
                "type":        ev_type,
                "from_tid":    tid_a,
                "to_tid":      tid_b,
                "from_team":   team_a,
                "to_team":     team_b,
                "dist_m":      round(dist_m, 2),
                "release_x_m": round(pos_a[0], 2),
                "release_y_m": round(pos_a[1], 2),
                "receive_x_m": round(pos_b[0], 2),
                "receive_y_m": round(pos_b[1], 2),
            }
            events.append(ev)
            print(f"  [PASE] frame={release_frame:4d}  "
                  f"T{team_a}#{tid_a} → T{team_b}#{tid_b}  "
                  f"{dist_m:.1f}m  [{ev_type}]")

        passes        = [e for e in events if e["type"] == "pass"]
        interceptions = [e for e in events if e["type"] == "interception"]
        pass_dists    = [e["dist_m"] for e in passes]
        stats = {
            "total_passes":        len(passes),
            "total_interceptions": len(interceptions),
            "avg_pass_m":  round(float(np.mean(pass_dists)), 2) if pass_dists else 0.0,
            "max_pass_m":  round(float(np.max(pass_dists)),  2) if pass_dists else 0.0,
            "min_pass_m":  round(float(np.min(pass_dists)),  2) if pass_dists else 0.0,
        }
        return events, stats


# ══════════════════════════════════════════════════════════════════════════════
# CSV enriquecido
# ══════════════════════════════════════════════════════════════════════════════

def enrich_csv(
    df: pd.DataFrame,
    pos_real_m: dict,
    speed_lookup: dict,
    dist_lookup: dict,
    possession_per_frame: dict,
    pass_events: list,
) -> pd.DataFrame:
    df = df.copy()
    df["x_m"]             = np.nan
    df["y_m"]             = np.nan
    df["speed_kmh"]       = np.nan
    df["distance_m"]      = np.nan
    df["team_possession"] = np.nan
    df["pass_type"]       = ""
    df["pass_dist_m"]     = np.nan

    pass_by_frame = {e["frame"]: e for e in pass_events}

    for idx, row in df.iterrows():
        f   = int(row["frame"])
        tid = int(row["track_id"])

        if f in pos_real_m and tid in pos_real_m[f]:
            xm, ym = pos_real_m[f][tid]
            df.at[idx, "x_m"] = round(xm, 3)
            df.at[idx, "y_m"] = round(ym, 3)

        if f in speed_lookup and tid in speed_lookup[f]:
            df.at[idx, "speed_kmh"]  = speed_lookup[f][tid]
            df.at[idx, "distance_m"] = dist_lookup.get(f, {}).get(tid, 0.0)

        if f in possession_per_frame:
            df.at[idx, "team_possession"] = possession_per_frame[f]

        if f in pass_by_frame and pass_by_frame[f]["from_tid"] == tid:
            ev = pass_by_frame[f]
            df.at[idx, "pass_type"]   = ev["type"]
            df.at[idx, "pass_dist_m"] = ev["dist_m"]

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Renderizado
# ══════════════════════════════════════════════════════════════════════════════

def render_analysis_video(
    tracked_video_path: Path,
    df: pd.DataFrame,
    possession_per_frame: dict,
    player_team: dict,
    pass_events: list,
    out_path: Path,
    fps: float,
    max_frames: int | None = None,
) -> None:
    """
    Genera vídeo de análisis sobre el tracked_video con:
      • Barra de posesión dinámica (sticky) en la parte superior
      • Contador de pases grande por equipo (T0 / T1)
      • Overlay de último pase/intercepción con fade-out 2s (texto grande)
    """
    cap = cv2.VideoCapture(str(tracked_video_path))
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (fw, fh))

    total = int(df["frame"].max()) + 1
    if max_frames is not None:
        total = min(total, max_frames)

    # Overlay de último pase
    PASS_SHOW_FRAMES = int(fps * 2.0)
    last_pass_info  = None
    last_pass_frame = -PASS_SHOW_FRAMES
    pass_by_frame   = {e["frame"]: e for e in pass_events}

    running_poss   = {0: 0, 1: 0}
    running_passes = {0: 0, 1: 0}
    font = cv2.FONT_HERSHEY_SIMPLEX

    print(f"  Resolución: {fw}×{fh}  |  {total} frames")

    frame_rel = 0
    while frame_rel < total:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Acumular posesión ─────────────────────────────────────────────────
        t_now = possession_per_frame.get(frame_rel, -1)
        if t_now in running_poss:
            running_poss[t_now] += 1

        # ── Barra de posesión ─────────────────────────────────────────────────
        BAR_H = 36
        BAR_Y = 8
        BAR_X = 8
        BAR_W = fw - 16
        total_p = running_poss[0] + running_poss[1]
        pct0 = running_poss[0] / max(total_p, 1)
        w0   = int(BAR_W * pct0)
        w1   = BAR_W - w0

        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (BAR_X - 4, BAR_Y - 4),
                      (BAR_X + BAR_W + 4, BAR_Y + BAR_H + 4),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        if w0 > 0:
            cv2.rectangle(frame, (BAR_X, BAR_Y),
                          (BAR_X + w0, BAR_Y + BAR_H), COLOR_TEAM0, -1)
        if w1 > 0:
            cv2.rectangle(frame, (BAR_X + w0, BAR_Y),
                          (BAR_X + BAR_W, BAR_Y + BAR_H), COLOR_TEAM1, -1)

        for txt, tx in [
            (f"{pct0 * 100:.0f}%", BAR_X + 8),
            (f"{(1 - pct0) * 100:.0f}%", BAR_X + w0 + 8),
        ]:
            cv2.putText(frame, txt, (tx, BAR_Y + BAR_H - 9),
                        font, 0.65, (0, 0, 0), 3)
            cv2.putText(frame, txt, (tx, BAR_Y + BAR_H - 9),
                        font, 0.65, (255, 255, 255), 1)

        label = "POSESION"
        (lw, _), _ = cv2.getTextSize(label, font, 0.45, 1)
        lx = BAR_X + BAR_W // 2 - lw // 2
        cv2.putText(frame, label, (lx, BAR_Y + BAR_H - 9),
                    font, 0.45, (0, 0, 0), 3)
        cv2.putText(frame, label, (lx, BAR_Y + BAR_H - 9),
                    font, 0.45, (220, 220, 220), 1)

        # ── Contador de pases ─────────────────────────────────────────────────
        if frame_rel in pass_by_frame:
            ev_now = pass_by_frame[frame_rel]
            if ev_now["type"] == "pass" and ev_now["from_team"] in running_passes:
                running_passes[ev_now["from_team"]] += 1
            last_pass_info  = ev_now
            last_pass_frame = frame_rel

        # ── Contador de pases (más grande) ───────────────────────────────────
        CNT_SCALE = 0.85
        CNTX = BAR_X
        CNTY = BAR_Y + BAR_H + 8
        (_, plh), _ = cv2.getTextSize("PASES", font, CNT_SCALE, 2)

        ov3 = frame.copy()
        base_w = cv2.getTextSize(
            f"PASES   T0: {running_passes[0]}    T1: {running_passes[1]}",
            font, CNT_SCALE, 2
        )[0][0]
        cv2.rectangle(ov3,
                      (CNTX - 6, CNTY - 4),
                      (CNTX + base_w + 10, CNTY + plh + 8),
                      (20, 20, 20), -1)
        cv2.addWeighted(ov3, 0.60, frame, 0.40, 0, frame)

        cv2.putText(frame, "PASES   T0: ", (CNTX, CNTY + plh),
                    font, CNT_SCALE, (0, 0, 0), 5)
        cv2.putText(frame, "PASES   T0: ", (CNTX, CNTY + plh),
                    font, CNT_SCALE, (200, 200, 200), 2)
        t0_x = CNTX + cv2.getTextSize("PASES   T0: ", font, CNT_SCALE, 2)[0][0]
        cv2.putText(frame, str(running_passes[0]), (t0_x, CNTY + plh),
                    font, CNT_SCALE, (0, 0, 0), 5)
        cv2.putText(frame, str(running_passes[0]), (t0_x, CNTY + plh),
                    font, CNT_SCALE, COLOR_TEAM0, 2)
        sep_x = t0_x + cv2.getTextSize(str(running_passes[0]), font, CNT_SCALE, 2)[0][0]
        cv2.putText(frame, "    T1: ", (sep_x, CNTY + plh),
                    font, CNT_SCALE, (0, 0, 0), 5)
        cv2.putText(frame, "    T1: ", (sep_x, CNTY + plh),
                    font, CNT_SCALE, (200, 200, 200), 2)
        t1_x = sep_x + cv2.getTextSize("    T1: ", font, CNT_SCALE, 2)[0][0]
        cv2.putText(frame, str(running_passes[1]), (t1_x, CNTY + plh),
                    font, CNT_SCALE, (0, 0, 0), 5)
        cv2.putText(frame, str(running_passes[1]), (t1_x, CNTY + plh),
                    font, CNT_SCALE, COLOR_TEAM1, 2)

        # ── Overlay de último pase (fade-out 2s, texto grande) ───────────────
        EV_SCALE = 1.1
        frames_since = frame_rel - last_pass_frame
        if last_pass_info is not None and frames_since < PASS_SHOW_FRAMES:
            ev       = last_pass_info
            alpha    = 1.0 - frames_since / PASS_SHOW_FRAMES
            color_ev = (0, 200, 80) if ev["type"] == "pass" else (0, 80, 220)
            label_ev = "PASE" if ev["type"] == "pass" else "INTERCEPCION"
            box_x = BAR_X
            box_y = CNTY + plh + 14
            (tw, th), _ = cv2.getTextSize(label_ev, font, EV_SCALE, 3)
            ov2 = frame.copy()
            cv2.rectangle(ov2,
                          (box_x - 6, box_y - 6),
                          (box_x + tw + 12, box_y + th + 10),
                          (20, 20, 20), -1)
            cv2.addWeighted(ov2, 0.65 * alpha, frame, 1.0 - 0.65 * alpha, 0, frame)
            cv2.putText(frame, label_ev, (box_x, box_y + th),
                        font, EV_SCALE, (0, 0, 0), 6)
            cv2.putText(frame, label_ev, (box_x, box_y + th),
                        font, EV_SCALE, color_ev, 3)

        writer.write(frame)
        frame_rel += 1
        if frame_rel % 300 == 0:
            p0 = running_poss[0] / max(total_p, 1) * 100
            p1 = running_poss[1] / max(total_p, 1) * 100
            print(f"  {frame_rel}/{total}  posesión T0={p0:.0f}% T1={p1:.0f}%")

    cap.release()
    writer.release()
    print(f"  Vídeo → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",          required=True)
    ap.add_argument("--csv",            required=True)
    ap.add_argument("--tracked-video",  default=None)
    ap.add_argument("--out-csv",        default="outputs/metrics/analysis.csv")
    ap.add_argument("--out-video",      default=None)
    ap.add_argument("--start-sec",      type=float, default=0.0)
    ap.add_argument("--max-frames",     type=int,   default=None)
    ap.add_argument("--kp-conf",        type=float, default=0.2)
    ap.add_argument("--frame-window",   type=int,   default=60)
    ap.add_argument("--max-dist-px",    type=float, default=60.0,
                    help="Umbral posesión en píxeles (default=60 para 1280×720)")
    ap.add_argument("--ball-gap",       type=int,   default=30,
                    help="Interpolación de balón: gap máximo a rellenar (frames)")
    ap.add_argument("--min-pass-m",     type=float, default=3.0)
    ap.add_argument("--min-pass-frames",type=int,   default=8)
    ap.add_argument("--kp-model",       default=None)
    args = ap.parse_args()

    kp_path = Path(args.kp_model) if args.kp_model else KP_MODEL_DYNAMIC
    if not kp_path.exists():
        print(f"ERROR: modelo keypoints no encontrado: {kp_path}")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    total_frames = int(df["frame"].max()) + 1
    if args.max_frames:
        total_frames = min(total_frames, args.max_frames)
        df = df[df["frame"] < total_frames].copy()

    _cap = cv2.VideoCapture(args.video)
    fps  = _cap.get(cv2.CAP_PROP_FPS) or 25.0
    _cap.release()

    print(f"\n{'='*64}")
    print(f"  Vídeo      : {args.video}")
    print(f"  CSV        : {args.csv}  ({len(df)} filas, {total_frames} frames)")
    print(f"  FPS        : {fps:.2f}   start-sec={args.start_sec}")
    print(f"  Ventana    : {args.frame_window}f = {args.frame_window/fps:.3f}s")
    print(f"  Posesión   : max_dist={args.max_dist_px}px  |  Ball gap: {args.ball_gap}f")
    print(f"  Pases      : min_poss={args.min_pass_frames}f  min_dist={args.min_pass_m}m")
    print(f"{'='*64}")

    # ── Team ID por voto mayoritario ─────────────────────────────────────────
    player_team = build_player_team(df)
    print(f"\n[Team] {sum(v==0 for v in player_team.values())} tracks T0  "
          f"{sum(v==1 for v in player_team.values())} tracks T1  "
          f"{sum(v==-1 for v in player_team.values())} tracks sin equipo")

    # ── Pass 1: homografía → pos_cm → pos_real_m ─────────────────────────────
    print("\n[Pass 1/3] Keypoints + posiciones reales (m)...")
    mapper = DynamicHomographyMapper(model_path=kp_path, conf=args.kp_conf)
    cap    = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_sec * fps))

    pos_cm: dict = {}
    kp_ok = 0

    for frame_rel in range(total_frames):
        ret, raw_frame = cap.read()
        if not ret:
            break

        n_kp = mapper.update(raw_frame)
        if n_kp >= 4:
            kp_ok += 1

        rows_f = df[df["frame"] == frame_rel]
        if rows_f.empty or not mapper.has_homography:
            continue

        feet = np.array(
            [[(r.x1 + r.x2) / 2, r.y2] for r in rows_f.itertuples()],
            dtype=np.float32,
        )
        xy_cm = mapper.transform_points(feet)

        pos_cm[frame_rel] = {}
        for i, row in enumerate(rows_f.itertuples()):
            if not np.isnan(xy_cm[i, 0]):
                pos_cm[frame_rel][int(row.track_id)] = (
                    float(xy_cm[i, 0]), float(xy_cm[i, 1])
                )

        if (frame_rel + 1) % 300 == 0:
            print(f"  {frame_rel+1}/{total_frames} — kp OK: {kp_ok/(frame_rel+1)*100:.0f}%")

    cap.release()
    print(f"  Keypoints ≥4: {kp_ok/max(total_frames,1)*100:.1f}%  "
          f"| Frames con pos: {len(pos_cm)}")

    pos_real_m = pos_cm_to_real_m(pos_cm)

    # ── Pass 2: Ball + Speed + Possession + Passes ────────────────────────────
    print("\n[Pass 2/3] Ball interpolation + Speed + Possession + Passes...")

    # Interpolación del balón
    ball_interp = BallInterpolator(max_gap=args.ball_gap)
    ball_px_per_frame, detected_frames = ball_interp.compute(df, total_frames)

    # Velocidad — suavizado de posiciones + comparativa antes/después
    player_ids = set(
        df[df["class"].isin(_PLAYER_CLASSES)]["track_id"].astype(int).unique()
    )

    # Suavizar posiciones para reducir jitter de homografía dinámica
    pos_smooth = smooth_positions(pos_real_m, window=7)

    # Velocidad RAW (sin suavizar) — solo para la comparativa
    _speed_raw, _ = SpeedEstimator(fps=fps, frame_window=args.frame_window).compute(
        pos_real_m, total_frames, player_ids
    )

    # Velocidad SUAVIZADA — la que se usa para todo
    speed_est = SpeedEstimator(fps=fps, frame_window=args.frame_window)
    speed_lookup, dist_lookup = speed_est.compute(pos_smooth, total_frames, player_ids)

    # Comparativa para los 3 tracks con más frames de velocidad
    frames_per_track: dict[int, int] = defaultdict(int)
    for fd in _speed_raw.values():
        for tid in fd:
            frames_per_track[tid] += 1
    top3 = sorted(frames_per_track, key=lambda t: frames_per_track[t], reverse=True)[:3]
    print(f"  [Smooth] Comparativa SIN vs CON suavizado (window=7):")
    print(f"    {'track':>6}  {'vmax_raw':>9}  {'vmax_smo':>9}  "
          f"{'vmed_raw':>9}  {'vmed_smo':>9}")
    for tid in top3:
        raw_s = [fd[tid] for fd in _speed_raw.values() if tid in fd]
        smo_s = [fd[tid] for fd in speed_lookup.values() if tid in fd]
        if not raw_s or not smo_s:
            continue
        print(f"    {tid:>6}  {max(raw_s):>8.1f}k  {max(smo_s):>8.1f}k  "
              f"  {float(np.median(raw_s)):>7.1f}k  {float(np.median(smo_s)):>7.1f}k")

    # Posesión (píxeles)
    poss_tracker = PossessionTrackerPixel(max_dist_px=args.max_dist_px)
    possession_per_frame, player_per_frame, _, pct = poss_tracker.compute(
        df, ball_px_per_frame, detected_frames, player_team, total_frames
    )

    # Pases
    pass_tracker = PassTracker(
        min_possession_frames=args.min_pass_frames,
        min_pass_m=args.min_pass_m,
    )
    pass_events, pass_stats = pass_tracker.compute(
        player_per_frame, pos_real_m, player_team, total_frames
    )

    # ── Resumen ───────────────────────────────────────────────────────────────
    print(f"\n  POSESIÓN (sobre frames con balón DETECTADO):")
    print(f"    Equipo 0 : {pct['team_0']:.1f}%")
    print(f"    Equipo 1 : {pct['team_1']:.1f}%")
    print(f"    Sin asignar: {pct['none']:.1f}%")

    print(f"\n  VELOCIDADES (top 5):")
    top: dict = defaultdict(float)
    for fd in speed_lookup.values():
        for tid, spd in fd.items():
            if spd > top[tid]:
                top[tid] = spd
    for tid, spd in sorted(top.items(), key=lambda x: x[1], reverse=True)[:5]:
        team = player_team.get(tid, -1)
        print(f"    track_id={tid:3d}  team={team}  v_max={spd:.1f} km/h")

    print(f"\n  PASES  : {pass_stats['total_passes']} "
          f"(media {pass_stats['avg_pass_m']:.1f}m, "
          f"max {pass_stats['max_pass_m']:.1f}m)")
    print(f"  INTERCEPCIONES: {pass_stats['total_interceptions']}")

    # ── Pass 3a: CSV ──────────────────────────────────────────────────────────
    print(f"\n[Pass 3/3] CSV enriquecido...")
    df_out = enrich_csv(
        df, pos_real_m, speed_lookup, dist_lookup, possession_per_frame, pass_events
    )
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"  CSV → {out_csv}  ({len(df_out)} filas)")

    # ── Pass 3b: Vídeo ────────────────────────────────────────────────────────
    if args.out_video:
        tracked_path = Path(args.tracked_video) if args.tracked_video else None
        if not tracked_path or not tracked_path.exists():
            print(f"\nAVISO: --tracked-video no encontrado, se omite el vídeo.")
        else:
            print(f"\nRenderizando vídeo...")
            render_analysis_video(
                tracked_video_path   = tracked_path,
                df                   = df,
                possession_per_frame = possession_per_frame,
                player_team          = player_team,
                pass_events          = pass_events,
                out_path             = Path(args.out_video),
                fps                  = fps,
                max_frames           = args.max_frames,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
