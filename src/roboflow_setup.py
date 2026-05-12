"""
Pipeline completo: descarga 3 proyectos forkeados de Roboflow, los fusiona
en un único dataset YOLOv8 y sube el resultado al proyecto football-tactical.

Uso: python src/roboflow_setup.py
"""

import os
import shutil
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

API_KEY  = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = "alfonso-secretara-tcnica"
TARGET_PROJECT = "football-tactical"

# Slugs reales (verificados via API)
SOURCE_PROJECTS = {
    "football-player-nnajt-68w3v":           "Football PLayer (805)",
    "football-player-jrjtj-urpir":            "football-player (664)",
    "football-players-detection-3zvbc-pcts8": "football-players-detection (372)",
    "football-ball-detection-rejhg-r1kak":    "football-ball-detection (1237)",
    "football-tracking-dyjjm-qweas":          "football-tracking (663)",
    "football-video-tracking-project-yjknb":  "football-video-tracking-project (663)",
}

# psg/bar son etiquetas de equipo en el dataset de 664 imgs → se normalizan a player
CLASS_MAP = {
    "player":     "player",
    "psg":        "player",
    "bar":        "player",
    "goalkeeper": "goalkeeper",
    "referee":    "referee",
    "ball":       "ball",
}

# football-tracking tiene data.yaml corrupto; las clases reales se deducen
# por distribución de etiquetas: 0=ball(473), 1=goalkeeper(1519), 2=player(13241), 3=referee(565)
CLASS_OVERRIDES = {
    "football-tracking-dyjjm-qweas": ["ball", "goalkeeper", "player", "referee"],
}
UNIFIED_CLASSES = ["player", "goalkeeper", "referee", "ball"]

RAW_DIR    = Path("data/dataset/raw")
MERGED_DIR = Path("data/dataset/merged")
SPLITS     = ["train", "valid", "test"]

# ── Paso 1: descarga ──────────────────────────────────────────────────────────

def download_project(ws, slug: str, name: str) -> Path:
    out_dir = RAW_DIR / slug
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"  [SKIP] Ya descargado: {name}")
        return out_dir

    print(f"  Exportando {name} en formato YOLOv8 (search_export)...")
    out_dir.mkdir(parents=True, exist_ok=True)
    ws.search_export(
        query="",
        format="yolov8",
        location=str(out_dir),
        dataset=slug,
    )
    print(f"  Descargado en {out_dir}")
    return out_dir


# ── Paso 2: fusión ────────────────────────────────────────────────────────────

def remap_label(src: Path, dst: Path, src_classes: list[str]) -> None:
    lines = src.read_text().strip().splitlines() if src.exists() else []
    remapped = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        src_name = src_classes[int(parts[0])].lower()
        unified  = CLASS_MAP.get(src_name)
        if unified is None:
            continue                      # descarta clases desconocidas
        remapped.append(f"{UNIFIED_CLASSES.index(unified)} " + " ".join(parts[1:]))
    dst.write_text("\n".join(remapped))


def merge_datasets() -> None:
    print("\n[MERGE] Fusionando datasets...")
    for split in SPLITS:
        (MERGED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    total = 0
    for slug in SOURCE_PROJECTS:
        yaml_files = list((RAW_DIR / slug).rglob("data.yaml"))
        if not yaml_files:
            print(f"  [WARN] Sin data.yaml en {slug}, omitido")
            continue

        meta       = yaml.safe_load(yaml_files[0].read_text())
        src_classes = CLASS_OVERRIDES.get(slug, meta.get("names", []))
        base       = yaml_files[0].parent
        prefix     = slug[:8]           # prefijo para evitar colisiones de nombre

        for split in SPLITS:
            img_dir = base / split / "images"
            lbl_dir = base / split / "labels"
            if not img_dir.exists():
                continue
            for img_path in img_dir.glob("*"):
                new_name = f"{prefix}_{img_path.name}"
                shutil.copy2(img_path, MERGED_DIR / split / "images" / new_name)
                remap_label(
                    lbl_dir / (img_path.stem + ".txt"),
                    MERGED_DIR / split / "labels" / (Path(new_name).stem + ".txt"),
                    src_classes,
                )
                total += 1

    yaml.dump(
        {
            "train": str(MERGED_DIR / "train" / "images"),
            "val":   str(MERGED_DIR / "valid" / "images"),
            "test":  str(MERGED_DIR / "test" / "images"),
            "nc":    len(UNIFIED_CLASSES),
            "names": UNIFIED_CLASSES,
        },
        open(MERGED_DIR / "data.yaml", "w"),
        default_flow_style=False,
    )
    print(f"  {total} imágenes fusionadas → {MERGED_DIR}")
    print(f"  Clases unificadas: {UNIFIED_CLASSES}")


# ── Paso 3: subida ────────────────────────────────────────────────────────────

def _upload_one(project, img_path: Path, lbl_path: Path, split: str) -> str:
    try:
        project.upload(
            image_path=str(img_path),
            annotation_path=str(lbl_path) if lbl_path.exists() and lbl_path.stat().st_size else None,
            annotation_labelmap=UNIFIED_CLASSES,
            split=split,
            num_retry_uploads=2,
            batch_name="merged_dataset",
        )
        return "ok"
    except Exception as e:
        return f"err:{img_path.name}:{e}"


def upload_to_target(rf) -> None:
    print(f"\n[UPLOAD] Subiendo a {TARGET_PROJECT}...")
    project = rf.workspace(WORKSPACE).project(TARGET_PROJECT)

    tasks = [
        (img, MERGED_DIR / split / "labels" / (img.stem + ".txt"), split)
        for split in SPLITS
        if (MERGED_DIR / split / "images").exists()
        for img in sorted((MERGED_DIR / split / "images").glob("*"))
    ]

    ok = err = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_upload_one, project, img, lbl, sp): img.name
                   for img, lbl, sp in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result == "ok":
                ok += 1
            else:
                err += 1
                print(f"  {result}")
            if i % 200 == 0:
                print(f"  Progreso: {i}/{len(tasks)} ({ok} ok, {err} err)")

    print(f"  Subida completada: {ok} ok, {err} errores")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    rf = Roboflow(api_key=API_KEY)
    ws = rf.workspace(WORKSPACE)

    print("=" * 60)
    print("PASO 1 — Descargar proyectos fuente")
    print("=" * 60)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for slug, name in SOURCE_PROJECTS.items():
        print(f"\n[{name}]")
        download_project(ws, slug, name)

    print("\n" + "=" * 60)
    print("PASO 2 — Fusionar datasets")
    print("=" * 60)
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    merge_datasets()

    print("\n" + "=" * 60)
    print("PASO 3 — Subir a football-tactical")
    print("=" * 60)
    upload_to_target(rf)

    print("\n✓ Pipeline completo.")


if __name__ == "__main__":
    main()
