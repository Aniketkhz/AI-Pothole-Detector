from ultralytics import YOLO
import os

# Load your trained model
model = YOLO('runs/detect/train17/weights/best.pt')

# Run detection on validation images
results = model.predict(
    source='data/images/val',
    save=True,  # Save results
    conf=0.5,   # Confidence threshold
    show=True   # Display results
)

print(f"\nDetection complete! Results saved in: {os.path.abspath('runs/detect/predict')}")