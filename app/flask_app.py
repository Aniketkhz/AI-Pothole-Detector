from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path

app = Flask(__name__)

# Configuration
BASE_DIR = Path(__file__).parent
app.config.update({
    'UPLOAD_FOLDER': BASE_DIR / 'static' / 'uploads',
    'RESULT_FOLDER': BASE_DIR / 'static' / 'results',
    'MODEL_URL': "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-pothole.pt",
    'MODEL_PATH': BASE_DIR / 'models' / 'pothole_detector.pt'
})

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# ====== MODEL SETUP ======
# Download model if missing
if not os.path.exists(app.config['MODEL_PATH']):
    print("Downloading pothole detection model...")
    os.makedirs(BASE_DIR / 'models', exist_ok=True)
    torch.hub.download_url_to_file(
        app.config['MODEL_URL'],
        str(app.config['MODEL_PATH'])
    )
    print("Download complete!")

# Load model
model = YOLO(str(app.config['MODEL_PATH']))
print("Model classes:", model.names)  # Should show {'0': 'pothole'}

# ====== ROUTES ======
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if 'image' not in request.files:
            return "No file selected", 400
            
        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400
            
        try:
            filename = secure_filename(file.filename)
            upload_path = app.config['UPLOAD_FOLDER'] / filename
            file.save(str(upload_path))
            
            # Process image with pothole detection
            img = cv2.imread(str(upload_path))
            results = model.predict(
                img,
                conf=0.5,  # Confidence threshold
                imgsz=640,  # Input size
                classes=[0],  # Only detect potholes
                augment=True  # Test-time augmentation
            )
            
            # Create annotated image
            annotated_img = results[0].plot(
                line_width=3,  # Thicker bounding boxes
                font_size=0.8,  # Larger labels
                labels=True,  # Show class labels
                pil=True  # Better rendering
            )
            
            # Save result
            result_path = app.config['RESULT_FOLDER'] / filename
            cv2.imwrite(str(result_path), annotated_img)
            
            print(f"Detected {len(results[0].boxes)} potholes")
            return render_template("result.html",
                                original_img=filename,
                                result_img=filename)
                                
        except Exception as e:
            print("Error:", e)
            return f"Error: {str(e)}", 500
            
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)