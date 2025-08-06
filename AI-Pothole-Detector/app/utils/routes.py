from flask import Flask, render_template, Response, jsonify, request
from detection.pothole_detector import PotholeDetector
import cv2
import os

app = Flask(__name__)
detector = PotholeDetector()

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

def generate_frames():
    camera = cv2.VideoCapture(0)  # Webcam
    while True:
        success, frame = camera.read()
        if not success: break
        
        frame, _ = detector.detect(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
