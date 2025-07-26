from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from pathlib import Path
from flask import send_from_directory

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
app.config.update({
    'UPLOAD_FOLDER': BASE_DIR / 'static' / 'uploads',
    'RESULT_FOLDER': BASE_DIR / 'static' / 'results'
})

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def detect_potholes(image_path):
    """Improved pothole detection with better filtering"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    
    
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    potholes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # More strict criteria
        if 500 < area < 50000:  # Adjusted size range
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if 0.2 < circularity < 1.2:  # Potholes are somewhat circular
                x,y,w,h = cv2.boundingRect(cnt)
                aspect_ratio = float(w)/h
                if 0.5 < aspect_ratio < 2:  # More square-like ratios
                    potholes.append(cnt)
    
    # Draw results
    for cnt in potholes:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        cv2.putText(img, f'Pothole {len(potholes)}', (x,y-10), 
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
            
        
            result_img, pothole_count = detect_potholes(str(upload_path))
            

            result_path = app.config['RESULT_FOLDER'] / filename
            cv2.imwrite(str(result_path), result_img)
            
            print(f"Detected {pothole_count} potential potholes")
            return render_template("result.html",
                                original_img=filename,
                                result_img=filename,
                                count=pothole_count)
                                
        except Exception as e:
            print("Error:", e)
            return f"Error: {str(e)}", 500
            
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)