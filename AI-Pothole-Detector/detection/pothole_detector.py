import cv2
from ultralytics import YOLO
import numpy as np

class PotholeDetector:
    def __init__(self, model_path='app/models/pothole_detector.pt'):
        self.model = YOLO(model_path)
        self.class_names = ['pothole']
    
    def detect(self, frame):
        results = self.model(frame)
        detections = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class_name': self.class_names[0]
                })
        
        annotated_frame = results[0].plot()
        return annotated_frame, detections
