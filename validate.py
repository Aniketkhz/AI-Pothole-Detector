from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/detect/pothole_training3/weights/best.pt')

# Run validation
results = model.val(
    data='data/dataset.yaml',
    split='val',        # Validate on validation set
    batch=8,            # Match your training batch size
    imgsz=640,          # Match your training image size
    conf=0.25,          # Confidence threshold
    iou=0.45,           # IoU threshold
    device='cpu',       # Use '0' for GPU if available
    name='pothole_val'  # Save results to 'runs/detect/pothole_val'
)