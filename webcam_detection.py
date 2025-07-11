import cv2
from ultralytics import YOLO

def main():
    # UNCOMMENT ONLY ONE OF THESE MODEL PATHS:
    
    # Option 1: Your trained pothole model (best choice)
    model_path = r"C:\Users\anike\AI-Pothole-Detector\runs\detect\pothole_training11\weights\best.pt"
    
    # Option 2: Pretrained yolov8n as fallback (will detect general objects)
    # model_path = "yolov8n.pt"
    
    # Load the model
    model = YOLO(model_path)
    
    # Webcam setup with Windows optimization
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't capture frame")
            break
            
        # Run detection with 50% confidence threshold
        results = model(frame, conf=0.5)
        
        # Display results with pothole count
        annotated = results[0].plot()
        pothole_count = len(results[0].boxes)
        cv2.putText(annotated, f"Potholes: {pothole_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Pothole Detection", annotated)
        
        # Exit on 'q' key
        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()