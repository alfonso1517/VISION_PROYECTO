# Proyecto: Visión Artificial aplicada al Fútbol
Proyecto Máster IA — Análisis táctico con cámara cenital

## Stack
- Python 3.10 en .venv (activar: source .venv/bin/activate)
- ultralytics (YOLOv8), supervision, opencv-python, ByteTrack
- matplotlib, numpy, Pillow, SoccerNet, yt-dlp

## Objetivo del proyecto
Pipeline completo de análisis táctico sobre vídeos de cámara táctica cenital:
1. Detección de jugadores, árbitro y balón con YOLOv8
2. Tracking multi-objeto con ByteTrack
3. Proyección homográfica al mapa 2D del campo
4. Generación de heatmaps, estimación de velocidad y zonas de influencia

## Fuentes de datos
- data/videos/ → clips .mp4 de cámara táctica descargados con yt-dlp
- data/frames/ → frames extraídos (1fps) para construir el dataset
- data/soccernet/ → JSONs de tracking de SoccerNet (ground truth para métricas)
- data/dataset/ → dataset anotado exportado desde Roboflow en formato YOLOv8

## Estructura del proyecto
- src/extract_frames.py → extrae frames de los vídeos
- src/train.py → fine-tuning YOLOv8 con el dataset de Roboflow
- src/track.py → detección + tracking sobre vídeo nuevo
- src/homography.py → proyección de coordenadas al campo 2D
- src/analysis.py → heatmaps, velocidad, zonas de influencia
- notebooks/ → exploración y visualizaciones
- models/weights/ → yolov8m.pt y checkpoints de entrenamiento
- outputs/ → vídeos anotados, heatmaps, métricas

## Fase actual: 1 — Obtención de datos
- [x] Acceso a SoccerNet concedido
- [x] Vídeos de cámara táctica localizados en YouTube
- [ ] Descargar vídeos con yt-dlp
- [ ] Instalar dependencias (requirements.txt)
- [ ] Extraer frames con extract_frames.py
- [ ] Descargar tracking data de SoccerNet

## Siguiente acción
Crear src/extract_frames.py

## Convenciones
- Comentarios y variables en inglés, commits en español
- Usar pathlib.Path en vez de os.path
- Guardar frames como .jpg calidad 95
- Resolución mínima de vídeo aceptada: 720p