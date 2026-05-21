"""
Clasificación de equipo por color de camiseta — método KMeans sobre píxeles.
Lógica de abdullahtarek/football_analysis (team_assigner/team_assigner.py):

  1. Por cada crop: tomar la mitad superior del bbox
  2. Filtrar píxeles verdes (césped) en HSV antes del KMeans
  3. KMeans(2) sobre los píxeles no-verdes
  4. Las 4 esquinas del crop identifican el cluster de fondo
  5. El color del jugador = centroide del cluster NON-fondo
  6. Fit global: KMeans(2) sobre todos los colores de jugador → 2 equipos
  7. Predicción cacheada por track_id

Interfaz idéntica a TeamClassifier para intercambiabilidad directa.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans

MAX_TEAM_SIZE = 13


class TeamClassifierKMeans:
    """
    Clasifica jugadores en dos equipos por color de camiseta (píxeles puros).

    Flujo recomendado:
      classifier.fit(crops)
      team_id = classifier.get_team(crop, tid)
      team_id = classifier.get_team_with_limit(crop, tid, frame_counts)
    """

    def __init__(self, **kwargs):
        self.kmeans     = None
        self._fitted    = False
        self._cache:     dict[int, int]        = {}  # track_id → team_id (permanente)
        self._emb_cache: dict[int, np.ndarray] = {}  # color 3D (compat. Re-ID)

    # ------------------------------------------------------------------
    def _get_player_color(self, crop: np.ndarray) -> np.ndarray:
        """
        Extrae el color representativo del jugador en el crop BGR.
        - Usa la mitad superior del crop (torso)
        - Filtra píxeles verdes (césped) en HSV antes del KMeans
        - KMeans(2) sobre los píxeles válidos
        - Las 4 esquinas determinan qué cluster es el fondo
        - Devuelve el centroide RGB del cluster de jugador
        """
        if crop.size == 0:
            return np.zeros(3)

        top_half = crop[: crop.shape[0] // 2, :]
        if top_half.size == 0:
            top_half = crop

        # Filtrar píxeles verdes (césped): H 35-85, S > 40 en HSV
        hsv = cv2.cvtColor(top_half, cv2.COLOR_BGR2HSV)
        green_mask = (
            (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) &
            (hsv[:, :, 1] > 40)
        )
        pixels_all = top_half.reshape(-1, 3).astype(np.float32)
        pixels = pixels_all[~green_mask.ravel()]
        if len(pixels) < 50:   # fallback si quedan muy pocos píxeles
            pixels = pixels_all
        if len(pixels) < 2:
            return pixels[0] if len(pixels) == 1 else np.zeros(3)

        km = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        km.fit(pixels)

        # Las 4 esquinas del top_half identifican el cluster de fondo
        h, w = top_half.shape[:2]
        corner_pixels = np.array([
            top_half[0, 0], top_half[0, -1],
            top_half[-1, 0], top_half[-1, -1],
        ], dtype=np.float32)
        corner_labels = km.predict(corner_pixels)
        bg_cluster = max(set(corner_labels.tolist()), key=corner_labels.tolist().count)
        player_cluster = 1 - bg_cluster

        return km.cluster_centers_[player_cluster]

    # ------------------------------------------------------------------
    def fit(self, crops: list) -> None:
        """Ajusta el KMeans global de equipos sobre una lista de crops BGR."""
        if len(crops) < 2:
            return
        print(f"  [TeamClassifierKMeans] Fit con {len(crops)} crops...")
        colors = np.array([self._get_player_color(c) for c in crops])
        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        self.kmeans.fit(colors)
        self._fitted = True
        print("  [TeamClassifierKMeans] Fit completado.")
        for i, center in enumerate(self.kmeans.cluster_centers_):
            r, g, b = int(center[2]), int(center[1]), int(center[0])  # BGR→RGB
            print(f"  [TeamClassifierKMeans] Cluster {i}: RGB=({r}, {g}, {b})")

    # ------------------------------------------------------------------
    def get_team(self, crop: np.ndarray, track_id: int) -> int:
        """Devuelve team_id (0 ó 1). Primera asignación permanente — no recalcula."""
        if track_id in self._cache:
            return self._cache[track_id]
        if not self._fitted:
            return 0

        color = self._get_player_color(crop)
        self._emb_cache[track_id] = color
        team_id = int(self.kmeans.predict(color.reshape(1, -1))[0])
        self._cache[track_id] = team_id
        return team_id

    # ------------------------------------------------------------------
    def get_team_with_limit(
        self,
        crop: np.ndarray,
        track_id: int,
        current_frame_counts: dict[int, int],
    ) -> int:
        """Igual que get_team pero con restricción de cardinalidad por frame."""
        team_id = self.get_team(crop, track_id)
        other   = 1 - team_id

        if current_frame_counts.get(team_id, 0) >= MAX_TEAM_SIZE:
            if current_frame_counts.get(other, 0) < MAX_TEAM_SIZE:
                return other

        return team_id
