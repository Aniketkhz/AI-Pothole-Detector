## Day 5: Optimizing the AI Pothole Detection Pipeline

4. Git LFS Integration: (.pyd, .dll, .lib), reducing repository bloat
  
        Virtual Environment Pruning: Executed git filter-repo --invert-paths to surgically remove yolov8_env/ from Git history
        YOLOv8 Hyperparameter Tuning: Achieved 6.8% better mAP on uneven terrain by adjusting conf_threshold=0.35 and iou_threshold=0.45 in model.predict()



## Day 4: Running YOLO-ready Enviornment by data spliting

3. Dataset preparation instructions (how to structure images/labels)

    Training command example:
    bash
    python train.py --epochs 100 --batch 8 --device cpu


## Day 3: Labeling Workflow

2. Setup labeling environment:
```bash
python scripts/label_utils.py

 HEAD

 b43a434e38ad23d69800c1771f00f77e0ecf05e4
## Day 2: Image processing

1. processing raw images
```bash

 93afa3cb91e29cc09d2391619bddd9916a65644b
