from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Configuration
BASE_DIR = Path(__file__).parent
app.config.update({
    'UPLOAD_FOLDER': BASE_DIR / 'static' / 'uploads',
    'RESULT_FOLDER': BASE_DIR / 'static' / 'results'
})

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def detect_potholes(image_path):
    """Basic pothole detection using image processing"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours that could be potholes
    potholes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 50000:  # Size range for potholes
            x,y,w,h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            if 0.2 < aspect_ratio < 5:  # Reasonable aspect ratios
                potholes.append(cnt)
    
    # Draw bounding boxes
    for cnt in potholes:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        cv2.putText(img, 'Pothole', (x,y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    
    return img, len(potholes)

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
            
            # Process image
            result_img, pothole_count = detect_potholes(str(upload_path))
            
            # Save result
            result_path = app.config['RESULT_FOLDER'] / filename
            cv2.imwrite(str(result_path), result_img)
            
            print(f"Detected {pothole_count} potential potholes")
            return render_template("result.html",
                                original_img=filename,
                                result_img=filename)
                                
        except Exception as e:
            print("Error:", e)
            return f"Error: {str(e)}", 500
            
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)