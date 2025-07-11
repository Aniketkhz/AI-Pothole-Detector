from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import os
import sys


os.environ["PYTHONUNBUFFERED"] = "1"  


def force_tqdm():
    import functools
    tqdm.__init__ = functools.partialmethod(tqdm.__init__, file=sys.stderr)

force_tqdm()


base_dir = Path(r"C:\Users\anike\AI-Pothole-Detector")
dataset_yaml = base_dir / "data" / "dataset.yaml"
weights = base_dir / "yolov8n.pt"

def train_model():
    model = YOLO(weights)
    results = model.train(
        data=str(dataset_yaml),
        epochs=100,
        imgsz=640,
        batch=8,
        name="pothole_training",
        device="cpu",
        verbose=True,  # Ensure verbose logging
    )
    print(f"Training complete! Model saved to: {results.save_dir}")

if __name__ == "__main__":
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Missing dataset.yaml at {dataset_yaml}")
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights at {weights}")
    
    train_model()