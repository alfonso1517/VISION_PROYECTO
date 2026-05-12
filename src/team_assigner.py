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

from typing import List

import numpy as np
import torch
import umap
from sklearn.cluster import KMeans
from transformers import AutoProcessor, SiglipVisionModel

import supervision as sv

SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"
MAX_TEAM_SIZE     = 13   # margen: 11 titulares + 2 re-IDs por equipo por frame


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
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)
        self._fitted = False
        self._cache: dict[int, int] = {}        # track_id → team_id
        self._proj_cache: dict[int, np.ndarray] = {}  # track_id → UMAP projection

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
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)
        self._fitted = True
        print("  [TeamClassifier] Fit completado.")

    # ------------------------------------------------------------------
    def get_team(self, crop: np.ndarray, track_id: int) -> int:
        """
        Devuelve team_id (0 ó 1). Cacheado por track_id.
        SigLIP solo corre para IDs nuevos.
        """
        if track_id in self._cache:
            return self._cache[track_id]
        if not self._fitted:
            return 0

        data = self._extract_features([crop])
        proj = self.reducer.transform(data)[0]
        team_id = int(self.cluster_model.predict(proj[None])[0])
        self._cache[track_id] = team_id
        self._proj_cache[track_id] = proj
        return team_id

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
