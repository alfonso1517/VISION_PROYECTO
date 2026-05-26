# Proyecto: Visión Artificial aplicada al Fútbol
Proyecto Máster IA — Análisis táctico con cámara broadcast

## Stack
- Python 3.12 en `.venv` (activar: `source .venv/bin/activate`)
- `ultralytics` (YOLOv8 + ONNX), `supervision`, `opencv-python`, `ByteTrack`
- `rfdetr` (RF-DETR detector), `transformers` (SigLIP para team assignment)
- `matplotlib`, `numpy`, `pandas`, `Pillow`

## Vídeos disponibles
| Archivo | Descripción | Start-sec relevante |
|---|---|---|
| `data/videos/tactico_01.mp4` | Vídeo largo (4.1GB) | — |
| `data/videos/tactico_02.mp4` | Madrid broadcast (2.1GB) | **3940** (t=65:40) |
| `data/videos/tactico_psg_bayern.mp4` | PSG vs Bayern (1.9GB, 59.8fps) | **1768** (t=29:28) |

## CSVs de tracks disponibles (outputs/metrics/)
| Archivo | Detector | Frames | Vídeo | Columnas |
|---|---|---|---|---|
| `FINAL_madrid_yolo_tracks.csv` | YOLOv8 | 0–999 | tactico_02 desde t=3940 | frame,track_id,class,team_id,x1,y1,x2,y2,confidence |
| `psg_rfdetr_500f_tracks.csv` | RF-DETR+SigLIP | 0–499 | tactico_psg_bayern desde t=1768 | ídem |
| `psg_yolo_500f_tracks.csv` | YOLOv8 | 0–499 | tactico_psg_bayern desde t=1768 | ídem |
| `FINAL_madrid_homography.csv` | YOLOv8 | 0–999 | Madrid | ídem + x_tac, y_tac |

## Estructura del proyecto
```
src/
  track.py           → YOLOv8 + ByteTrack + SigLIP team assignment → CSV + vídeo anotado
  track_rfdetr.py    → RF-DETR + ByteTrack + SigLIP team assignment → CSV + vídeo anotado
  homography.py      → Proyección 2D dinámica (YOLO Pose keypoints) + minimap + side-by-side
  analysis.py        → (PENDIENTE) Speed estimator + ball possession
  extract_frames.py  → Extrae frames de vídeos

models/weights/
  football_pitch_kp.onnx   → Roboflow football-field-detection-f07vi/15 (267MB, YOLO Pose 32 kp)
  field_keypoints.pt        → Modelo hmzbo (portrait, no usar con broadcast)
  yolov8m.pt / best.pt      → Detector de jugadores

assets/
  tactical_map.jpg           → Mapa táctico hmzbo (landscape tras rotate 90°CW)
  pitch_map_labels.json      → Keypoints hmzbo (obsoleto)

outputs/tracked_videos/
  FINAL_madrid_yolo.mp4      → Tracked Madrid (YOLOv8, 1000f, con bboxes y colores equipo)
  FINAL_psg_rfdetr.mp4       → Tracked PSG (RF-DETR, 500f, con bboxes y colores equipo)
  madrid_comparativa.mp4     → Side-by-side: tracked Madrid | minimap 2D (1000f, suavizado)
  radar_psg_smooth_100f.mp4  → PSG minimap overlay 100f con suavizado temporal
```

## Convenciones
- Comentarios y variables en inglés, commits en español
- Usar `pathlib.Path` en vez de `os.path`
- `--start-sec` siempre necesario para alinear CSV con vídeo

---

## Estado actual del pipeline (mayo 2026)

### ✅ Completado

#### Tracking
- `track.py`: YOLOv8 + ByteTrack + SigLIP → CSV con `team_id` (0/1)
- `track_rfdetr.py`: RF-DETR + ByteTrack + SigLIP → CSV idéntico
- Ambos guardan CSV automáticamente en `outputs/metrics/`
- **RF-DETR funciona mejor en PSG** (más detecciones, mejor precisión)

#### Homografía dinámica (`src/homography.py`)
- **Modo `--mode dynamic`** (default, recomendado):
  - Modelo `football_pitch_kp.onnx` (Roboflow f07vi) detecta 32 keypoints del campo por frame
  - `findHomography(frame_px → cm_reales)` recalculado por frame → maneja cámara broadcast que panea
  - `SoccerPitchConfiguration`: campo 12000×7000 cm, 32 vértices nombrados
  - **MSE check**: si keypoints se mueven < 25px² entre frames consecutivos, reutiliza H
  - **Suavizado temporal**: media ponderada de las últimas 5 matrices H (pesos: 0.4, 0.15, 0.15, 0.15, 0.15) → elimina parpadeo
  - 100% keypoints en Madrid (conf=0.3), 100% en PSG (conf=0.2, start-sec=1768)
- **Modo `--mode static`**: UI interactiva manual (para cámara fija cenital)
- **Modo side-by-side** (`--tracked-video`): genera vídeo 1889×540 con tracked a la izq y minimap 2D grande a la dcha

#### Vídeo comparativa Madrid
```bash
python src/homography.py \
  --video data/videos/tactico_02.mp4 \
  --csv outputs/metrics/FINAL_madrid_yolo_tracks.csv \
  --out-video outputs/tracked_videos/madrid_comparativa.mp4 \
  --tracked-video outputs/tracked_videos/FINAL_madrid_yolo.mp4 \
  --start-sec 3940 --mode dynamic --kp-conf 0.3
```

---

### 🔜 PRÓXIMO PASO: `src/analysis.py`

Implementar speed estimator y ball possession para el **vídeo del PSG**.

#### Referencia: repositorio abdullahtarek/football_analysis
- YouTube: https://youtu.be/neBZ6huolkg (t=3:06:25 camera movement, 3:41:50 perspective, 4:05:40 speed)
- GitHub: https://github.com/abdullahtarek/football_analysis

#### Por qué NO necesitamos su Camera Movement Estimator
El tutorial usa optical flow (Lucas-Kanade) para compensar el movimiento de cámara ANTES de aplicar homografía estática. **Nosotros ya lo tenemos resuelto mejor**: nuestra homografía dinámica (YOLO Pose 32kp por frame) ya da coordenadas en cm del mundo real, compensando el panning automáticamente.

#### Lo que SÍ implementamos: Speed & Distance Estimator

**Lógica del tutorial adaptada a nuestro stack:**

```python
# Su lógica (frame_window=5, fps=24):
distance_m  = euclidean(pos_m_frame_N, pos_m_frame_N+5)
time_s      = 5 / 24
speed_kmh   = (distance_m / time_s) * 3.6

# Nuestra adaptación (PSG: fps=59.8, usar frame_window=15 o 30):
# - Sus posiciones vienen de ViewTransformer estático (metros)
# - Las nuestras vienen de DynamicHomographyMapper → cm → /100 → metros
# - Con fps=59.8 y window=5: mides cada 0.084s → demasiado ruidoso
# - Con fps=59.8 y window=15: mides cada 0.25s → más estable
# - Con fps=59.8 y window=30: mides cada 0.5s → suave, realista
```

#### Ball Possession Tracker

```python
# Por cada frame donde hay balón detectado:
# 1. Obtener posición del balón en cm (homografía)
# 2. Para cada jugador en ese frame: calcular distancia al balón en cm
# 3. Si distancia < umbral (ej: 200cm = 2m): ese jugador tiene posesión
# 4. Acumular frames_possession[team_id]
# 5. % posesión = frames_possession[0] / total_frames_con_balon * 100
```

#### Plan de implementación `src/analysis.py`

```python
class SpeedEstimator:
    def __init__(self, fps: float, frame_window: int = 30):
        ...
    def compute(self, df: pd.DataFrame, mapper) -> pd.DataFrame:
        # Añade columnas: x_m, y_m, speed_kmh, distance_total_m

class PossessionTracker:
    def __init__(self, max_dist_cm: float = 200.0):
        ...
    def compute(self, df: pd.DataFrame, mapper) -> dict:
        # Devuelve: {"team_0": 45.2, "team_1": 54.8, "none": 0.0}

def render_analysis_video(
    video_path, tracked_video_path, df_enriched,
    possession_stats, mapper, out_path, start_sec, ...
):
    # Side-by-side con:
    # - Izquierda: tracked video con speed/km·h sobre cada jugador
    # - Derecha: minimap 2D con barra de posesión en la parte superior
```

#### Comando objetivo para PSG

```bash
# Primero generar el CSV enriquecido:
python src/analysis.py \
  --video data/videos/tactico_psg_bayern.mp4 \
  --csv outputs/metrics/psg_rfdetr_500f_tracks.csv \
  --out-csv outputs/metrics/psg_rfdetr_500f_analysis.csv \
  --out-video outputs/tracked_videos/psg_analysis.mp4 \
  --tracked-video outputs/tracked_videos/FINAL_psg_rfdetr.mp4 \
  --start-sec 1768 --kp-conf 0.2 --frame-window 30
```

---

## Comandos de referencia

### Tracking PSG con RF-DETR
```bash
source .venv/bin/activate
python src/track_rfdetr.py \
  --input data/videos/tactico_psg_bayern.mp4 \
  --output outputs/tracked_videos/FINAL_psg_rfdetr.mp4 \
  --start-sec 1768 --max-frames 500
```

### Homografía dinámica PSG (minimap overlay)
```bash
python src/homography.py \
  --video data/videos/tactico_psg_bayern.mp4 \
  --csv outputs/metrics/psg_rfdetr_500f_tracks.csv \
  --out-video outputs/tracked_videos/radar_psg_smooth.mp4 \
  --start-sec 1768 --mode dynamic --kp-conf 0.2
```

### Homografía dinámica PSG (side-by-side)
```bash
python src/homography.py \
  --video data/videos/tactico_psg_bayern.mp4 \
  --csv outputs/metrics/psg_rfdetr_500f_tracks.csv \
  --out-video outputs/tracked_videos/psg_comparativa.mp4 \
  --tracked-video outputs/tracked_videos/FINAL_psg_rfdetr.mp4 \
  --start-sec 1768 --mode dynamic --kp-conf 0.2
```
