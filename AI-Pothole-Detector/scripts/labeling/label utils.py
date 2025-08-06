import os
import shutil
from pathlib import Path
import yaml

def prepare_labeling_env():
    """Organize preprocessed data for labeling"""

    labeled_dir = Path("data/labeled")
    (labeled_dir/"images").mkdir(parents=True, exist_ok=True)
    (labeled_dir/"labels").mkdir(exist_ok=True)
    
    processed_images = list(Path("data/processed/images").glob("*.*"))
    val_count = int(len(processed_images) * 0.2)
    
    for i, img_path in enumerate(processed_images):
        dest = labeled_dir/"images"/("val" if i < val_count else "train")/img_path.name
        shutil.copy(img_path, dest)

def create_yolo_config():
    
    """Generate dataset.yaml for YOLOv8"""
    config = {
        'path': str(Path("data/labeled").absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'pothole'}
    }
    
    with open("data/labeled/dataset.yaml", 'w') as f:
        yaml.dump(config, f)

def verify_labels():
    """Check label-file consistency"""
    missing = []
    for img in Path("data/labeled/images/train").glob("*.*"):
        label = Path("data/labeled/labels/train")/f"{img.stem}.txt"
        if not label.exists():
            missing.append(img.name)
    
    if missing:
        print(f"Warning: {len(missing)} images missing labels")
        return False
    return True

if __name__ == "__main__":
    prepare_labeling_env()
    create_yolo_config()
    print("Labeling environment ready! Run: labelImg data/labeled/images/train data/labeled/labels/train")