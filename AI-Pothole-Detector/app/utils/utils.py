import numpy as np

def non_max_suppression(prediction, conf_thres=0.5, iou_thres=0.5):
    """Non-Maximum Suppression with enhanced filtering"""
    output = [None for _ in range(len(prediction))]
    
    for image_i, pred in enumerate(prediction):
        # Filter by confidence
        pred = pred[pred[:, 4] >= conf_thres]
        
        if not pred.shape[0]:
            continue
            
        # Compute (x1, y1, x2, y2) and area
        box = pred[:, :4]
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        
        # Sort by confidence (descending)
        order = pred[:, 4].argsort()[::-1]
        pred = pred[order]
        
        keep = []
        while pred.shape[0]:
            # Get the most confident prediction
            keep.append(pred[0])
            
            if pred.shape[0] == 1:
                break
                
            # Get IOUs for all boxes
            ious = bbox_iou(pred[0:1, :4], pred[1:, :4], area[order][1:])
            
            # Remove overlapping boxes
            pred = pred[1:][ious < iou_thres]
            order = order[1:][ious < iou_thres]
            
        output[image_i] = torch.stack(keep) if keep else None
        
    return output

def bbox_iou(box1, box2, area2):
    """Calculates IoU while considering area for efficiency"""
    inter_x1 = np.maximum(box1[0], box2[:, 0])
    inter_y1 = np.maximum(box1[1], box2[:, 1])
    inter_x2 = np.minimum(box1[2], box2[:, 2])
    inter_y2 = np.minimum(box1[3], box2[:, 3])
    
    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
    union_area = area2 + (box1[2] - box1[0]) * (box1[3] - box1[1]) - inter_area
    
    return inter_area / union_area