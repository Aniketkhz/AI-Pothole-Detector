from detection.pothole_detector import PotholeDetector
import cv2

detector = PotholeDetector()
test_img = cv2.imread("data/images/train/istockphoto-1013829164-612x612.jpg")  # Use your image
result, detections = detector.detect(test_img)

print(f"Found {len(detections)} potholes")
cv2.imwrite("detection_result.jpg", result)
