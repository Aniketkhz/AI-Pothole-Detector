import os
from flask import Flask, render_template, redirect, url_for, Response
from datetime import datetime
from collections import defaultdict

# Initialize Flask app
app = Flask(__name__)

# Sample data
detection_history = [
    {
        'filename': 'sample1.jpg',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'potholes': [{'confidence': 0.95, 'x': 100, 'y': 200, 'width': 50, 'height': 50}],
        'image_path': 'static/uploads/sample1.jpg'
    }
]

@app.route('/')
def home():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    stats = defaultdict(int)
    total_potholes = sum(len(d['potholes']) for d in detection_history)
    
    for detection in detection_history:
        if detection['potholes']:
            stats['detected'] += 1
        else:
            stats['no_detections'] += 1
    
    return render_template(
        'dashboard.html',
        total_detections=total_potholes,
        images_processed=len(detection_history),
        avg_potholes=total_potholes / max(1, len(detection_history)),
        recent_detections=detection_history[-5:][::-1],
        stats=stats
    )

@app.route('/upload')
def upload():
    return render_template('upload.html')

# Add this if you need video feed
@app.route('/video_feed')
def video_feed():
    # Implement your actual video feed logic here
    return Response("Video feed placeholder", mimetype='text/plain')

if __name__ == '__main__':
    # Print debug information
    print("Template folder:", app.template_folder)
    print("Static folder:", app.static_folder)
    app.run(debug=True)