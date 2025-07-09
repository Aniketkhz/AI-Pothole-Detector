from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data=r'/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    device='cpu'  # Change to '0' if you have GPU
)
