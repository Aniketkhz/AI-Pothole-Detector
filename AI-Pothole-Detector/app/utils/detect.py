import cv2
import torch
import numpy as np
from models import Darknet
from utils.utils import non_max_suppression, scale_coords
from utils.datasets import LoadImages

def detect(source, weights, img_size=416, conf_thres=0.5, iou_thres=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = Darknet('config/yolov3.cfg', img_size).to(device)
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    model.eval()
    
    # Set Dataloader
    dataset = LoadImages(source, img_size=img_size)
    
    # Run inference
    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            pred = model(img)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        
        # Process detections
        detections = []
        for i, det in enumerate(pred):
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                
                # Filter by size and aspect ratio
                for *xyxy, conf, cls in det:
                    w = xyxy[2] - xyxy[0]
                    h = xyxy[3] - xyxy[1]
                    area = w * h
                    aspect_ratio = w / h
                    
                    # Pothole-specific filters
                    if area > 500 and 0.5 < aspect_ratio < 2.0:
                        detections.append([*xyxy, conf.item()])
        
        return detections