from ultralytics import YOLO

model = YOLO("runs/detect/pothole_training/weights/best.pt")  # Update path
results = model.predict("test_image.jpg", save=True)