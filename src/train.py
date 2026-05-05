"""
Fine-tuning YOLOv8m sobre el dataset fusionado de football-tactical.
Si valid está vacío, hace split 90/10 automáticamente desde train.

Uso: python src/train.py [--epochs N] [--batch N] [--imgsz N]
"""

import argparse
import random
import shutil
import yaml
from pathlib import Path

from ultralytics import YOLO

DATASET_DIR  = Path("data/dataset/merged")
WEIGHTS_DIR  = Path("models/weights")
OUTPUTS_DIR  = Path("outputs")
BASE_MODEL   = "yolov8m.pt"
VAL_FRACTION = 0.15


def ensure_val_split(dataset_dir: Path) -> None:
    train_imgs = list((dataset_dir / "train" / "images").glob("*"))
    valid_imgs = list((dataset_dir / "valid" / "images").glob("*"))

    if valid_imgs:
        return

    print(f"Valid vacío — haciendo split {int((1-VAL_FRACTION)*100)}/{int(VAL_FRACTION*100)} desde train...")
    (dataset_dir / "valid" / "images").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "valid" / "labels").mkdir(parents=True, exist_ok=True)

    random.seed(42)
    val_imgs = random.sample(train_imgs, k=int(len(train_imgs) * VAL_FRACTION))

    for img in val_imgs:
        lbl = dataset_dir / "train" / "labels" / (img.stem + ".txt")
        shutil.move(str(img), dataset_dir / "valid" / "images" / img.name)
        if lbl.exists():
            shutil.move(str(lbl), dataset_dir / "valid" / "labels" / lbl.name)

    print(f"  Train: {len(train_imgs) - len(val_imgs)} imgs | Valid: {len(val_imgs)} imgs")


def build_data_yaml(dataset_dir: Path) -> Path:
    yaml_path = dataset_dir / "data.yaml"
    meta = yaml.safe_load(yaml_path.read_text())
    # Roboflow exporta paths relativos; los convertimos a absolutos para evitar
    # errores según desde qué directorio se lanza el entrenamiento
    meta["train"] = str((dataset_dir / "train" / "images").resolve())
    meta["val"]   = str((dataset_dir / "valid" / "images").resolve())
    meta["test"]  = str((dataset_dir / "test"  / "images").resolve())
    abs_yaml = dataset_dir / "data_abs.yaml"
    yaml.dump(meta, open(abs_yaml, "w"), default_flow_style=False)
    return abs_yaml


def train(epochs: int, batch: int, imgsz: int) -> None:
    ensure_val_split(DATASET_DIR)
    data_yaml = build_data_yaml(DATASET_DIR)

    model = YOLO(BASE_MODEL)   # descarga yolov8m.pt si no existe

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(OUTPUTS_DIR),
        name="football_v1",
        exist_ok=True,
        device="0",            # GPU; cambia a "cpu" si no hay CUDA
        patience=20,           # early stopping
        save=True,
        save_period=10,        # checkpoint cada 10 épocas
        val=True,
        plots=True,
        verbose=True,
    )

    # Copia best.pt a models/weights/ para fácil acceso
    best_src = Path(results.save_dir) / "weights" / "best.pt"
    if best_src.exists():
        dest = WEIGHTS_DIR / "football_v1_best.pt"
        shutil.copy2(best_src, dest)
        print(f"\nMejor checkpoint guardado en {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--imgsz",  type=int, default=640)
    args = parser.parse_args()
    train(args.epochs, args.batch, args.imgsz)
