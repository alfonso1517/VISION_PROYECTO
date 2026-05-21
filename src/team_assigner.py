"""
Clasificación de equipo por color de camiseta.
Lógica extraída de roboflow/sports (sports/common/team.py):
  SigLIP (embeddings visuales) → UMAP (reducción 3D) → KMeans (2 clusters)

Estrategia de caché:
  - Fit inicial sobre los jugadores del primer frame con ≥4 detecciones (~22 crops)
  - Predicción cacheada por track_id: SigLIP sólo procesa crops nuevos
  - Coste real: ~22 crops en fit + 1 crop por track_id nuevo durante el vídeo

Restricción de cardinalidad:
  - Máximo MAX_TEAM_SIZE IDs por equipo en cada frame
  - El jugador más dudoso (mayor distancia al centroide KMeans) se redirige al otro equipo
  - Árbitros y porteros no cuentan para el límite (se gestionan en track.py)
"""

from collections import deque
from typing import List

import numpy as np
import torch
import umap
from sklearn.cluster import KMeans
from transformers import AutoProcessor, SiglipVisionModel

import supervision as sv

SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"
MAX_TEAM_SIZE     = 13   # margen: 11 titulares + 2 re-IDs por equipo por frame
HISTORY_LEN       = 10   # frames de historial para suavizado temporal


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class TeamClassifier:
    """
    Clasifica jugadores en dos equipos:
      SigLIP → embeddings 768D → UMAP 3D → KMeans(2)

    Flujo recomendado:
      classifier.fit(crops)
      team_id = classifier.get_team(crop, tid)
      team_id = classifier.get_team_with_limit(crop, tid, frame_counts)
    """

    def __init__(self, device: str | None = None, batch_size: int = 32):
        self.device = device or _get_device()
        self.batch_size = batch_size
        print(f"  [TeamClassifier] Cargando SigLIP en {self.device}...")
        self.features_model = SiglipVisionModel.from_pretrained(
            SIGLIP_MODEL_PATH
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = None   # se instancia en fit() con n_neighbors adaptado al nº de muestras
        self.cluster_model = KMeans(n_clusters=2)
        self._fitted       = False
        self._dist_p75     = float("inf")                    # umbral de confianza del cluster
        self._cache:          dict[int, int]        = {}     # track_id → equipo actual (mayoría)
        self._proj_cache:     dict[int, np.ndarray] = {}     # track_id → UMAP projection 3D
        self._emb_cache:      dict[int, np.ndarray] = {}     # track_id → SigLIP embedding 768D
        self._history:        dict[int, deque]      = {}     # track_id → últimas N asignaciones
        self._last_confident: dict[int, int]        = {}     # track_id → última asig. confiable

    # ------------------------------------------------------------------
    def _extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        pil_crops = [sv.cv2_to_pillow(c) for c in crops]
        data = []
        with torch.no_grad():
            for start in range(0, len(pil_crops), self.batch_size):
                batch = pil_crops[start : start + self.batch_size]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                emb = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(emb)
        return np.concatenate(data)

    def _distance_to_centroid(self, proj: np.ndarray, team_id: int) -> float:
        """Distancia euclídea en espacio UMAP entre una proyección y el centroide del equipo."""
        center = self.cluster_model.cluster_centers_[team_id]
        return float(np.linalg.norm(proj - center))

    # ------------------------------------------------------------------
    def fit(self, crops: List[np.ndarray]) -> None:
        """Ajusta UMAP + KMeans sobre una lista de crops BGR de jugadores."""
        if len(crops) < 2:
            return
        print(f"  [TeamClassifier] Fit con {len(crops)} crops...")
        data = self._extract_features(crops)
        n_neighbors = min(len(crops) - 1, 15)
        self.reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, random_state=42)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

        # Percentil 75 de distancias al centroide → umbral de confianza del cluster
        distances = [
            float(np.linalg.norm(proj - self.cluster_model.cluster_centers_[label]))
            for proj, label in zip(projections, self.cluster_model.labels_)
        ]
        self._dist_p75 = float(np.percentile(distances, 75))

        self._fitted = True
        print(f"  [TeamClassifier] Fit completado. Umbral confianza p75={self._dist_p75:.4f}")

    # ------------------------------------------------------------------
    def get_team(self, crop: np.ndarray, track_id: int) -> int:
        """
        Devuelve team_id (0 ó 1) con confianza de cluster y suavizado temporal.
        SigLIP solo corre para track_ids nuevos; el resto usa la proyección cacheada.
        """
        if not self._fitted:
            return 0

        # SigLIP + UMAP: solo para track_ids nuevos (caro)
        if track_id not in self._proj_cache:
            data = self._extract_features([crop])
            self._emb_cache[track_id] = data[0]
            proj = self.reducer.transform(data)[0]
            self._proj_cache[track_id] = proj

        proj     = self._proj_cache[track_id]
        raw_team = int(self.cluster_model.predict(proj[None])[0])
        dist     = float(np.linalg.norm(proj - self.cluster_model.cluster_centers_[raw_team]))

        # ── Confianza del cluster ─────────────────────────────────────────────
        if dist > self._dist_p75 and track_id in self._last_confident:
            vote = self._last_confident[track_id]
        else:
            vote = raw_team
            self._last_confident[track_id] = raw_team

        # ── Suavizado temporal por mayoría ────────────────────────────────────
        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=HISTORY_LEN)
        self._history[track_id].append(vote)

        majority = max(set(self._history[track_id]), key=self._history[track_id].count)
        self._cache[track_id] = majority
        return majority

    # ------------------------------------------------------------------
    def get_team_with_limit(
        self,
        crop: np.ndarray,
        track_id: int,
        current_frame_counts: dict[int, int],
    ) -> int:
        """
        Igual que get_team pero aplica restricción de cardinalidad por frame.

        Si el equipo asignado ya tiene MAX_TEAM_SIZE jugadores en este frame,
        el jugador dudoso (mayor distancia al centroide) se redirige al otro equipo.

        Args:
            current_frame_counts: {team_id: nº jugadores ya asignados este frame}
        """
        team_id = self.get_team(crop, track_id)
        other   = 1 - team_id

        if current_frame_counts.get(team_id, 0) >= MAX_TEAM_SIZE:
            # Solo redirigir si el otro equipo aún tiene hueco
            if current_frame_counts.get(other, 0) < MAX_TEAM_SIZE:
                return other
            # Ambos llenos: confirmar la predicción original (caso muy raro)

        return team_id
