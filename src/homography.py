"""
Proyección homográfica frame → mapa táctico 2D.

Dos modos:
  --mode dynamic  (para cámara broadcast que panea)
      Detecta 32 keypoints del campo con YOLOv8-Pose en cada frame.
      Recalcula findHomography frame a frame → coordenadas en cm reales.
      Modelo: models/weights/football_pitch_kp.onnx
      (football-field-detection-f07vi de Roboflow Universe)

  --mode static   (para cámara fija cenital)
      UI interactiva manual: vídeo | mapa táctico side-by-side.
      Un único findHomography para todo el vídeo.

Inspirado en:
  - roboflow/sports (github.com/roboflow/sports)
  - hmzbo/Football-Analytics-with-Deep-Learning-and-Computer-Vision

Uso (dinámico — recomendado para broadcast):
  python src/homography.py \\
    --video data/videos/tactico_02.mp4 \\
    --csv outputs/metrics/FINAL_madrid_yolo_tracks.csv \\
    --out-video outputs/tracked_videos/radar_madrid.mp4 \\
    --start-sec 3940 --max-frames 100 --mode dynamic

Uso (estático — cámara fija):
  python src/homography.py \\
    --video data/videos/tactico_02.mp4 \\
    --csv outputs/metrics/FINAL_madrid_yolo_tracks.csv \\
    --out-csv outputs/metrics/FINAL_madrid_homography.csv \\
    --out-video outputs/tracked_videos/radar_madrid.mp4 \\
    --start-sec 3940 --mode static
"""

import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────────────────────
_SRC_DIR       = Path(__file__).parent
_ASSET_DIR     = _SRC_DIR.parent / "assets"
_MODELS_DIR    = _SRC_DIR.parent / "models" / "weights"

TAC_MAP_PATH   = _ASSET_DIR  / "tactical_map.jpg"
KEY_JSON_PATH  = _ASSET_DIR  / "pitch_map_labels.json"

# Modelos de keypoints
KP_MODEL_DYNAMIC = _MODELS_DIR / "football_pitch_kp.onnx"   # Roboflow f07vi
KP_MODEL_HMZBO   = _MODELS_DIR / "field_keypoints.pt"       # hmzbo (portrait)

# ── SoccerPitchConfiguration (de roboflow/sports/configs/soccer.py) ───────────
@dataclass
class SoccerPitchConfiguration:
    """Dimensiones y vértices del campo en cm. Fuente: roboflow/sports."""
    width:                  int = 7000    # cm
    length:                 int = 12000   # cm
    penalty_box_width:      int = 4100
    penalty_box_length:     int = 2015
    goal_box_width:         int = 1832
    goal_box_length:        int = 550
    centre_circle_radius:   int = 915
    penalty_spot_distance:  int = 1100

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        W, L = self.width, self.length
        pbw, pbl = self.penalty_box_width, self.penalty_box_length
        gbw, gbl = self.goal_box_width,    self.goal_box_length
        psd        = self.penalty_spot_distance
        ccr        = self.centre_circle_radius
        return [
            (0, 0),                                       # 0
            (0, (W - pbw) / 2),                           # 1
            (0, (W - gbw) / 2),                           # 2
            (0, (W + gbw) / 2),                           # 3
            (0, (W + pbw) / 2),                           # 4
            (0, W),                                       # 5
            (gbl, (W - gbw) / 2),                         # 6
            (gbl, (W + gbw) / 2),                         # 7
            (psd, W / 2),                                 # 8
            (pbl, (W - pbw) / 2),                         # 9
            (pbl, (W - gbw) / 2),                         # 10
            (pbl, (W + gbw) / 2),                         # 11
            (pbl, (W + pbw) / 2),                         # 12
            (L / 2, 0),                                   # 13
            (L / 2, W / 2 - ccr),                         # 14
            (L / 2, W / 2 + ccr),                         # 15
            (L / 2, W),                                   # 16
            (L - pbl, (W - pbw) / 2),                     # 17
            (L - pbl, (W - gbw) / 2),                     # 18
            (L - pbl, (W + gbw) / 2),                     # 19
            (L - pbl, (W + pbw) / 2),                     # 20
            (L - psd, W / 2),                             # 21
            (L - gbl, (W - gbw) / 2),                     # 22
            (L - gbl, (W + gbw) / 2),                     # 23
            (L, 0),                                       # 24
            (L, (W - pbw) / 2),                           # 25
            (L, (W - gbw) / 2),                           # 26
            (L, (W + gbw) / 2),                           # 27
            (L, (W + pbw) / 2),                           # 28
            (L, W),                                       # 29
            (L / 2 - ccr, W / 2),                         # 30
            (L / 2 + ccr, W / 2),                         # 31
        ]

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
        (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
        (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
        (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
        (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30),
    ])


SOCCER_CFG     = SoccerPitchConfiguration()
PITCH_VERTICES = np.array(SOCCER_CFG.vertices, dtype=np.float32)   # (32, 2) en cm

# ── Colores (BGR) ─────────────────────────────────────────────────────────────
COLOR_TEAM0      = (50, 205,  50)
COLOR_TEAM1      = (50,  50, 220)
COLOR_BALL       = ( 0, 230, 230)
COLOR_REF        = ( 0, 165, 255)
COLOR_DARK       = (20,  20,  20)
COLOR_PITCH_BG   = (34, 139,  34)
COLOR_PITCH_LINE = (255, 255, 255)


# ── Dibujo del campo en cm (basado en roboflow/sports/annotators/soccer.py) ───

def _draw_pitch_cm(mm_w: int, mm_h: int, padding: int = 10) -> np.ndarray:
    """
    Dibuja el campo de fútbol en tamaño (mm_w × mm_h) usando
    SoccerPitchConfiguration (unidades en cm).
    """
    cfg   = SOCCER_CFG
    sx    = (mm_w - 2 * padding) / cfg.length
    sy    = (mm_h - 2 * padding) / cfg.width

    img   = np.full((mm_h, mm_w, 3), COLOR_PITCH_BG, dtype=np.uint8)

    def cm2px(xc, yc):
        return int(xc * sx) + padding, int(yc * sy) + padding

    # Líneas de las aristas
    for s, e in cfg.edges:
        p1 = cm2px(*cfg.vertices[s - 1])
        p2 = cm2px(*cfg.vertices[e - 1])
        cv2.line(img, p1, p2, COLOR_PITCH_LINE, 1)

    # Círculo central
    cx, cy = cm2px(cfg.length / 2, cfg.width / 2)
    r = int(cfg.centre_circle_radius * min(sx, sy))
    cv2.circle(img, (cx, cy), r, COLOR_PITCH_LINE, 1)

    # Puntos de penalti
    for xc in [cfg.penalty_spot_distance, cfg.length - cfg.penalty_spot_distance]:
        cv2.circle(img, cm2px(xc, cfg.width / 2), 3, COLOR_PITCH_LINE, -1)

    return img


def _players_on_pitch_cm(
    img: np.ndarray,
    rows: pd.DataFrame,
    mapper,
    mm_w: int,
    mm_h: int,
    padding: int = 10,
) -> np.ndarray:
    """Dibuja jugadores en el campo (coords en cm) sobre 'img'."""
    cfg = SOCCER_CFG
    sx  = (mm_w - 2 * padding) / cfg.length
    sy  = (mm_h - 2 * padding) / cfg.width

    feet = np.array(
        [[(r.x1 + r.x2) / 2, r.y2] for r in rows.itertuples()],
        dtype=np.float32,
    )
    xy_cm = mapper.transform_points(feet)

    for i, row in enumerate(rows.itertuples()):
        xc, yc = float(xy_cm[i, 0]), float(xy_cm[i, 1])
        if np.isnan(xc):
            continue
        if not (-500 < xc < cfg.length + 500 and -500 < yc < cfg.width + 500):
            continue

        px = int(np.clip(xc * sx + padding, 2, mm_w - 3))
        py = int(np.clip(yc * sy + padding, 2, mm_h - 3))

        cls = str(row.clss)
        if cls == "ball":
            color, r = COLOR_BALL, 4
        elif cls == "referee":
            color, r = COLOR_REF, 4
        else:
            color = COLOR_TEAM0 if int(getattr(row, "team_id", 0)) == 0 else COLOR_TEAM1
            r = 5

        cv2.circle(img, (px, py), r,     (0, 0, 0), -1)
        cv2.circle(img, (px, py), r - 1, color,     -1)

    return img


# ── Carga del mapa táctico (modo estático) ────────────────────────────────────

def load_tactical_map() -> np.ndarray:
    """Carga tactical_map.jpg y lo rota 90° CW → landscape (546×314)."""
    tac = cv2.imread(str(TAC_MAP_PATH))
    if tac is None:
        raise FileNotFoundError(f"No se encontró: {TAC_MAP_PATH}")
    return cv2.rotate(tac, cv2.ROTATE_90_CLOCKWISE)


# ── Homografía dinámica — modelo YOLOv8-Pose (Roboflow f07vi) ─────────────────

class DynamicHomographyMapper:
    """
    Detecta los 32 keypoints del campo en cada frame con YOLOv8-Pose.
    Recalcula findHomography frame→cm solo cuando los keypoints se desplazan.

    Lógica idéntica a roboflow/sports/examples/soccer/main.py::render_radar.
    """

    def __init__(
        self,
        model_path: Path,
        conf: float = 0.30,
        displacement_tol: float = 25.0,
    ):
        from ultralytics import YOLO
        # task='pose' necesario para ONNX sin metadata
        self.model = YOLO(str(model_path), task="pose")
        self.conf  = conf
        self.tol   = displacement_tol        # umbral MSE en px²
        self.H: np.ndarray | None = None
        self._prev_kp:   np.ndarray | None = None   # (32, 2) frame anterior
        self._prev_mask: np.ndarray | None = None   # (32,) bool
        # Suavizado temporal: media ponderada de las últimas 5 matrices H
        self._H_history: deque = deque(maxlen=5)
        self._H_weights = [0.4, 0.15, 0.15, 0.15, 0.15]   # más reciente primero

    @property
    def has_homography(self) -> bool:
        return len(self._H_history) > 0

    def _smoothed_H(self) -> np.ndarray | None:
        """
        Media ponderada de las últimas matrices H almacenadas en _H_history.
        Peso 0.4 al frame más reciente, 0.15 a cada uno de los 4 anteriores.
        Normaliza los pesos si hay menos de 5 frames en historia.
        """
        n = len(self._H_history)
        if n == 0:
            return None
        if n == 1:
            return self._H_history[0]
        w = np.array(self._H_weights[:n], dtype=np.float64)
        w /= w.sum()          # renormalizar si hay menos de 5 frames
        H_avg = np.zeros((3, 3), dtype=np.float64)
        for i, H in enumerate(reversed(self._H_history)):  # más reciente = índice 0
            H_avg += w[i] * H.astype(np.float64)
        return H_avg

    def update(self, frame: np.ndarray) -> int:
        """
        Infiere keypoints, filtra visibles y actualiza H si hay movimiento.
        Devuelve número de keypoints visibles detectados.
        """
        result  = self.model(frame, conf=self.conf, verbose=False)[0]
        if result.keypoints is None or len(result.keypoints.xy) == 0:
            return 0

        kp_xy = result.keypoints.xy[0].cpu().numpy()   # (32, 2) en px del frame
        mask  = (kp_xy[:, 0] > 1) & (kp_xy[:, 1] > 1)
        n_vis = int(mask.sum())

        if n_vis < 4:
            return n_vis

        src = kp_xy[mask].astype(np.float32)        # px frame  (n, 2)
        dst = PITCH_VERTICES[mask]                   # cm campo  (n, 2)

        # ── ¿Recalcular H? MSE de keypoints comunes con frame anterior ────────
        if self.H is not None and self._prev_mask is not None:
            common = mask & self._prev_mask
            if common.sum() >= 4:
                mse = float(np.mean(
                    (kp_xy[common] - self._prev_kp[common]) ** 2
                ))
                if mse <= self.tol:
                    # Cámara no se ha movido significativamente → reutiliza H
                    self._prev_kp   = kp_xy.copy()
                    self._prev_mask = mask.copy()
                    return n_vis

        H, _ = cv2.findHomography(src, dst)
        if H is not None:
            self.H = H
            self._H_history.append(H)   # añadir a historia para suavizado

        self._prev_kp   = kp_xy.copy()
        self._prev_mask = mask.copy()
        return n_vis

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        H = self._smoothed_H()
        if H is None or points.size == 0:
            return np.full_like(points, np.nan)
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


# ── Homografía estática — puntos manuales ─────────────────────────────────────

class ViewTransformer:
    """Homografía fija calculada a partir de puntos calibrados manualmente."""

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.H, _ = cv2.findHomography(
            source.astype(np.float32), target.astype(np.float32)
        )
        if self.H is None:
            raise ValueError("findHomography devolvió None. Puntos colineales o < 4.")

    @property
    def has_homography(self) -> bool:
        return True

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)


# ── UI de calibración (modo estático) ─────────────────────────────────────────

def select_points_interactively(
    video_path: str,
    start_sec: float,
    tac_map: np.ndarray,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """
    UI lado a lado: vídeo (izq) | mapa táctico landscape (dcha).
    Navega con ← / → (±50 frames). ENTER confirma (≥ 4 pares). Z deshace.
    """
    cap     = cv2.VideoCapture(video_path)
    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    JUMP    = 50
    tac_h, tac_w = tac_map.shape[:2]

    def _read_frame(idx: int) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(idx, total_f - 1)))
        ret, f = cap.read()
        return f if ret else np.zeros((1080, 1920, 3), dtype=np.uint8)

    sample = _read_frame(int(start_sec * fps_vid))
    vid_h, vid_w = sample.shape[:2]
    VID_PW = 760
    VID_PH = int(VID_PW * vid_h / vid_w)
    vid_sx, vid_sy = VID_PW / vid_w, VID_PH / vid_h
    TAC_PH = VID_PH
    TAC_PW = int(tac_w * TAC_PH / tac_h)
    tac_scaled = cv2.resize(tac_map, (TAC_PW, TAC_PH))
    tac_sx, tac_sy = TAC_PW / tac_w, TAC_PH / tac_h
    SEP = 4
    canvas_w, canvas_h = VID_PW + SEP + TAC_PW, VID_PH + 50

    cur_idx = [int(start_sec * fps_vid)]
    pixel_pts: list = []
    tac_pts:   list = []
    state   = [0]
    pending = [None]
    dirty   = [True]

    def _rebuild() -> np.ndarray:
        vid_s = cv2.resize(_read_frame(cur_idx[0]), (VID_PW, VID_PH))
        c = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        c[:VID_PH, :VID_PW]             = vid_s
        c[:VID_PH, VID_PW:VID_PW + SEP] = 40
        c[:VID_PH, VID_PW + SEP:]        = tac_scaled
        col = (0, 200, 255)
        for i in range(len(tac_pts)):
            vx = int(pixel_pts[i][0] * vid_sx)
            vy = int(pixel_pts[i][1] * vid_sy)
            tx = int(tac_pts[i][0]   * tac_sx) + VID_PW + SEP
            ty = int(tac_pts[i][1]   * tac_sy)
            for cx, cy in [(vx, vy), (tx, ty)]:
                cv2.circle(c, (cx, cy), 7, (0, 0, 0), -1)
                cv2.circle(c, (cx, cy), 6, col, -1)
                cv2.putText(c, str(i + 1), (cx + 8, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        n = len(tac_pts)
        hdr = ("Par {}: clic en VIDEO (izq)".format(n + 1) if state[0] == 0
               else "Par {}: clic en MAPA (dcha)".format(n + 1))
        border = ((0, 0), (VID_PW - 1, VID_PH - 1)) if state[0] == 0 else \
                 ((VID_PW + SEP, 0), (canvas_w - 1, VID_PH - 1))
        cv2.rectangle(c, border[0], border[1], (0, 200, 255), 3)
        n_ok = "OK" if n >= 4 else f"faltan {4-n}"
        cv2.putText(c, f"{hdr}  |  Pares: {n} ({n_ok})  |  "
                    f"Frame {cur_idx[0]} (t={cur_idx[0]/fps_vid:.1f}s)",
                    (8, VID_PH + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)
        cv2.putText(c, "<- -> navegar  |  Z deshacer  |  ENTER confirmar  |  ESC salir",
                    (8, VID_PH + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)
        return c

    window = "Calibracion homografia"

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state[0] == 0 and x < VID_PW and y < VID_PH:
            pending[0] = (int(x / vid_sx), int(y / vid_sy))
            state[0] = 1; dirty[0] = True
            print(f"  Par {len(tac_pts)+1} — video ({pending[0][0]},{pending[0][1]}) → clic mapa")
        elif state[0] == 1 and x > VID_PW + SEP and y < VID_PH:
            tac_x = (x - VID_PW - SEP) / tac_sx
            tac_y = y / tac_sy
            pixel_pts.append(pending[0]); tac_pts.append((tac_x, tac_y))
            state[0] = 0; pending[0] = None; dirty[0] = True
            print(f"         mapa ({tac_x:.1f}, {tac_y:.1f}) — par {len(tac_pts)} OK")

    initial = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, initial)
    cv2.waitKey(1)
    cv2.setMouseCallback(window, on_click)

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32) and len(tac_pts) >= 4:
            break
        elif key in (13, 32):
            print(f"  Mínimo 4 pares, tienes {len(tac_pts)}.")
        elif key == 27:
            cap.release(); cv2.destroyAllWindows(); sys.exit(0)
        elif key in (81, 2):
            cur_idx[0] = max(0, cur_idx[0] - JUMP); state[0] = 0
            pending[0] = None; dirty[0] = True
        elif key in (83, 3):
            cur_idx[0] = min(total_f - 1, cur_idx[0] + JUMP); state[0] = 0
            pending[0] = None; dirty[0] = True
        elif key in (ord('z'), ord('Z')) and pixel_pts:
            pixel_pts.pop(); tac_pts.pop(); state[0] = 0
            pending[0] = None; dirty[0] = True
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            if len(tac_pts) >= 4:
                break
            cap.release(); sys.exit(1)
        if dirty[0]:
            dirty[0] = False; cv2.imshow(window, _rebuild())

    cap.release(); cv2.destroyAllWindows()
    print(f"  {len(tac_pts)} pares confirmados.")
    return pixel_pts, np.array(tac_pts, dtype=np.float32)


# ── Persistencia de puntos (modo estático) ────────────────────────────────────

def save_points(path: Path, pixel: list, tac: np.ndarray) -> None:
    data = {"pixel": [list(p) for p in pixel], "tac_map": tac.tolist()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    print(f"  Puntos guardados en {path}")


def load_points(path: Path) -> tuple[list, np.ndarray]:
    data  = json.loads(path.read_text())
    pixel = [tuple(p) for p in data["pixel"]]
    if "tac_map" in data:
        tac = np.array(data["tac_map"], dtype=np.float32)
    elif "real" in data:
        print(f"  AVISO: {path} usa formato antiguo (metros). Elimínalo y recalibra.")
        sys.exit(1)
    else:
        raise KeyError("JSON sin clave 'tac_map'.")
    print(f"  {len(pixel)} puntos cargados desde {path}")
    return pixel, tac


# ── Lectura de frame ──────────────────────────────────────────────────────────

def get_frame_at_sec(video_path: str, start_sec: float) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"No se pudo leer frame en t={start_sec}s de {video_path}")
    return frame


# ── Procesado del CSV (solo modo estático) ────────────────────────────────────

def process_csv(
    csv_path: Path,
    mapper: ViewTransformer,
    out_path: Path,
    tac_shape: tuple[int, int],
) -> pd.DataFrame:
    df   = pd.read_csv(csv_path)
    tac_h, tac_w = tac_shape
    feet = np.array(
        [[(r.x1 + r.x2) / 2, r.y2] for r in df.itertuples()], dtype=np.float32
    )
    xy      = mapper.transform_points(feet)
    df["x_tac"] = np.round(xy[:, 0], 1)
    df["y_tac"]  = np.round(xy[:, 1], 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  CSV guardado en {out_path}")
    print(f"  X_tac: [{df['x_tac'].min():.0f}, {df['x_tac'].max():.0f}]  (mapa 0–{tac_w})")
    print(f"  Y_tac: [{df['y_tac'].min():.0f}, {df['y_tac'].max():.0f}]  (mapa 0–{tac_h})")
    return df


# ── Renderizado del vídeo ─────────────────────────────────────────────────────

def render_video(
    video_path: str,
    df: pd.DataFrame,
    mapper,
    out_path: Path,
    start_sec: float = 0.0,
    dynamic: bool = True,
    max_frames: int | None = None,
    tac_map: np.ndarray | None = None,      # solo modo static
    tracked_video_path: Path | None = None, # modo side-by-side
) -> None:
    """
    Renderiza el vídeo con minimap.

    Si tracked_video_path se indica, genera un vídeo side-by-side:
      [vídeo trackeado (con bboxes) | minimap 2D grande]
    en lugar de superponer el minimap en esquina sobre el vídeo original.
    """
    cap  = cv2.VideoCapture(video_path)
    fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))

    # ── Modo side-by-side ─────────────────────────────────────────────────────
    side_by_side = tracked_video_path is not None
    if side_by_side:
        tracked_cap = cv2.VideoCapture(str(tracked_video_path))
        # Panel izquierdo: vídeo trackeado escalado a altura fija
        PANEL_H = 540
        left_w  = int(fw * PANEL_H / fh)   # mantiene AR del vídeo original
        # Panel derecho: minimap 2D del campo (AR = length/width = 12000/7000)
        right_w = int(PANEL_H * SOCCER_CFG.length / SOCCER_CFG.width)
        out_w   = left_w + 4 + right_w     # separador de 4px
        out_h   = PANEL_H
        # Minimap ocupa todo el panel derecho
        mm_w, mm_h = right_w, PANEL_H
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
        print(f"  Modo side-by-side: {left_w}×{PANEL_H} | {right_w}×{PANEL_H}  → {out_w}×{out_h}")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (fw, fh))
        # Minimap de esquina: 22% del alto del frame
        mm_h = int(fh * 0.22)
        if dynamic:
            mm_w = int(mm_h * SOCCER_CFG.length / SOCCER_CFG.width)
        else:
            tac_h_orig, tac_w_orig = tac_map.shape[:2]
            mm_w = int(mm_h * tac_w_orig / tac_h_orig)

    df_idx = df.copy()
    df_idx.columns = [c if c != "class" else "clss" for c in df_idx.columns]
    by_frame = {f: grp for f, grp in df_idx.groupby("frame")}
    total    = int(df_idx["frame"].max()) + 1
    if max_frames is not None:
        total = min(total, max_frames)

    mode_str = "dinámico (keypoints YOLO-Pose)" if dynamic else "estático (mapa táctico)"
    print(f"  Modo: {mode_str}")
    print(f"  Minimap: {mm_w}×{mm_h}px  |  Renderizando {total} frames...")

    kp_ok = 0
    frame_rel = 0
    while frame_rel < total:
        ret, frame = cap.read()
        if not ret:
            break

        rows = by_frame.get(frame_rel, pd.DataFrame())

        # ── Inferencia de keypoints y construcción del minimap ────────────────
        if dynamic:
            n_kp = mapper.update(frame)
            kp_ok += int(n_kp >= 4)
            minimap = _draw_pitch_cm(mm_w, mm_h)
            if not rows.empty and mapper.has_homography:
                minimap = _players_on_pitch_cm(minimap, rows, mapper, mm_w, mm_h)
        else:
            minimap = cv2.resize(tac_map, (mm_w, mm_h), interpolation=cv2.INTER_AREA)
            if not rows.empty and mapper.has_homography:
                tac_h_o, tac_w_o = tac_map.shape[:2]
                sx, sy = mm_w / tac_w_o, mm_h / tac_h_o
                feet = np.array(
                    [[(r.x1 + r.x2) / 2, r.y2] for r in rows.itertuples()],
                    dtype=np.float32,
                )
                xy_tac = mapper.transform_points(feet)
                for i, row in enumerate(rows.itertuples()):
                    xt, yt = float(xy_tac[i, 0]), float(xy_tac[i, 1])
                    if np.isnan(xt) or not (-20 < xt < tac_w_o + 20 and -20 < yt < tac_h_o + 20):
                        continue
                    px = int(np.clip(xt * sx, 2, mm_w - 3))
                    py = int(np.clip(yt * sy, 2, mm_h - 3))
                    cls = str(row.clss)
                    if cls == "ball":
                        color, r = COLOR_BALL, 4
                    elif cls == "referee":
                        color, r = COLOR_REF, 4
                    else:
                        color = COLOR_TEAM0 if int(getattr(row, "team_id", 0)) == 0 else COLOR_TEAM1
                        r = 5
                    cv2.circle(minimap, (px, py), r,     (0, 0, 0), -1)
                    cv2.circle(minimap, (px, py), r - 1, color,     -1)

        # ── Composición del frame de salida ───────────────────────────────────
        if side_by_side:
            # Leer frame del vídeo trackeado
            ret_t, tracked_frame = tracked_cap.read()
            if not ret_t:
                tracked_frame = np.zeros((fh, fw, 3), dtype=np.uint8)
            left_panel  = cv2.resize(tracked_frame, (left_w, PANEL_H))
            separator   = np.full((PANEL_H, 4, 3), COLOR_DARK, dtype=np.uint8)
            composite   = np.hstack([left_panel, separator, minimap])
            writer.write(composite)
        else:
            # Minimap en esquina inferior derecha del frame original
            margin = 8
            x0 = fw - mm_w - margin
            y0 = fh - mm_h - margin
            frame[y0 - 2: y0 + mm_h + 2, x0 - 2: x0 + mm_w + 2] = COLOR_DARK
            frame[y0: y0 + mm_h, x0: x0 + mm_w] = minimap
            writer.write(frame)

        frame_rel += 1
        if frame_rel % 200 == 0:
            pct = f" — kp OK en {kp_ok/frame_rel*100:.0f}% frames" if dynamic else ""
            print(f"  {frame_rel}/{total}{pct}")

    cap.release()
    if side_by_side:
        tracked_cap.release()
    writer.release()
    if dynamic and frame_rel > 0:
        print(f"  Keypoints detectados (≥4) en {kp_ok/frame_rel*100:.1f}% de frames")
    print(f"  Vídeo guardado en {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Homografía frame → campo 2D + minimap en vídeo"
    )
    ap.add_argument("--video",      required=True)
    ap.add_argument("--csv",        required=True)
    ap.add_argument("--out-csv",    default="outputs/metrics/tracks_tac.csv")
    ap.add_argument("--out-video",  default=None)
    ap.add_argument("--start-sec",  type=float, default=0.0)
    ap.add_argument("--max-frames", type=int,   default=None)
    ap.add_argument("--mode",       choices=["dynamic", "static"], default="dynamic")
    ap.add_argument("--kp-model",   default=None,
                    help="Ruta al modelo de keypoints .pt/.onnx  "
                         "(default: models/weights/football_pitch_kp.onnx)")
    ap.add_argument("--kp-conf",    type=float, default=0.30,
                    help="Confianza mínima para detección de keypoints (modo dynamic)")
    ap.add_argument("--points",        default="outputs/homography_points.json")
    ap.add_argument("--tracked-video", default=None,
                    help="Vídeo trackeado (con bboxes) para modo side-by-side con el minimap 2D")
    args = ap.parse_args()

    # ── 1. Construir el mapper ────────────────────────────────────────────────
    tac_map = None
    if args.mode == "dynamic":
        kp_path = Path(args.kp_model) if args.kp_model else KP_MODEL_DYNAMIC
        if not kp_path.exists():
            print(f"ERROR: No se encontró el modelo de keypoints: {kp_path}")
            print("Descarga football-field-detection-f07vi de Roboflow Universe")
            print("y guárdalo en models/weights/football_pitch_kp.onnx")
            sys.exit(1)
        print(f"\nModo DINÁMICO — modelo: {kp_path.name}")
        print(f"  Confianza: {args.kp_conf}  |  32 keypoints × SoccerPitchConfiguration")
        mapper = DynamicHomographyMapper(model_path=kp_path, conf=args.kp_conf)

    else:
        tac_map = load_tactical_map()
        print(f"\nModo ESTÁTICO — mapa táctico: {tac_map.shape[1]}×{tac_map.shape[0]}px")
        points_path = Path(args.points)
        if points_path.exists():
            pixel_pts, tac_pts = load_points(points_path)
        else:
            print(f"Abriendo UI de calibración en t={args.start_sec}s...")
            pixel_pts, tac_pts = select_points_interactively(
                args.video, args.start_sec, tac_map
            )
            save_points(points_path, pixel_pts, tac_pts)
        src    = np.array(pixel_pts, dtype=np.float32)
        mapper = ViewTransformer(source=src, target=tac_pts)
        print(f"  Homografía calculada con {len(pixel_pts)} puntos.")

        if args.out_csv:
            df_base = pd.read_csv(args.csv)
            print(f"\nProcesando CSV: {args.csv}")
            process_csv(Path(args.csv), mapper, Path(args.out_csv), tac_map.shape[:2])

    # ── 2. Renderizar vídeo ───────────────────────────────────────────────────
    if args.out_video:
        df = pd.read_csv(args.csv)
        print(f"\nRenderizando: {args.out_video}")
        tracked_path = Path(args.tracked_video) if args.tracked_video else None
        if tracked_path and not tracked_path.exists():
            print(f"AVISO: --tracked-video no encontrado: {tracked_path}  (se ignora)")
            tracked_path = None
        render_video(
            video_path         = args.video,
            df                 = df,
            mapper             = mapper,
            out_path           = Path(args.out_video),
            start_sec          = args.start_sec,
            dynamic            = (args.mode == "dynamic"),
            max_frames         = args.max_frames,
            tac_map            = tac_map,
            tracked_video_path = tracked_path,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
