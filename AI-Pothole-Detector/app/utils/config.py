import os
from pathlib import Path

# Absolute paths configuration
class Config:
    BASE_DIR = Path(r"C:\Users\anike\AI-Pothole-Detector")
    DATASET_YAML = BASE_DIR / "data" / "dataset.yaml"
    WEIGHTS = BASE_DIR / "yolov8n.pt"
    
    @staticmethod
    def verify():
        """Verify all paths exist"""
        required = {
            "Dataset YAML": Config.DATASET_YAML,
            "Model weights": Config.WEIGHTS,
            "Training images": Config.BASE_DIR / "data" / "images" / "train",
            "Validation images": Config.BASE_DIR / "data" / "images" / "val"
        }
        
        print("\n=== PATH VERIFICATION ===")
        for name, path in required.items():
            exists = path.exists()
            print(f"{name+':':<16} {path} {'✅' if exists else '❌'}")
            if not exists:
                raise FileNotFoundError(f"Missing path: {path}")