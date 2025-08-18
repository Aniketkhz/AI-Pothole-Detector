from flask import Flask, render_template, request
import os

app = Flask(__name__, template_folder='../../templates')  # Adjusted path
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template('result.html', 
                            filename=filename, 
                            pothole_count=3)  # Example value

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html',
                        total_potholes=42,
                        recent_detections=[
                            {"date": "2023-10-01", "count": 5},
                            {"date": "2023-09-30", "count": 2}
                        ])

if __name__ == '__main__':
    app.run(debug=True)