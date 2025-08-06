import os
from pathlib import Path

# Configuration
PATHS_TO_CHECK = [
    r"C:\Users\anike\AI-Pothole-Detector\data\dataset.yaml",
    r"C:\Users\anike\AI-Pothole-Detector\yolov8n.pt",
    r"C:\Users\anike\AI-Pothole-Detector\data\images\train",
    r"C:\Users\anike\AI-Pothole-Detector\data\images\val"
]

print("=== SYSTEM DIAGNOSTICS ===")
print(f"Current directory: {os.getcwd()}")
print(f"Python version: {os.sys.version}")

for path in PATHS_TO_CHECK:
    exists = os.path.exists(path)
    print(f"\nChecking: {path}")
    print(f"Exists: {'✅' if exists else '❌'}")
    if exists:
        if path.endswith('.yaml'):
            with open(path, 'r') as f:
                print("Contents:", f.read().strip())
        elif os.path.isdir(path):
            print(f"Files count: {len(os.listdir(path))}")