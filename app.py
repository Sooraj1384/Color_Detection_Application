from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os

from color_detection.detection import detect_colors
from color_detection.utils import allowed_file

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'static', 'processed')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(original_path)
        try:
            output_image, color_summary = detect_colors(original_path)
            output_filename = "processed_" + filename
            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
            # Save processed image
            output_image.save(processed_path)

            return render_template('result.html',
                                   filename=output_filename,
                                   color_summary=color_summary)
        except Exception as e:
            flash(f"Error processing image: {e}")
            return redirect(url_for('index'))
    else:
        flash('Allowed image types are png, jpg, jpeg, gif')
        return redirect(url_for('index'))

@app.route('/live')
def live_detection():
    return render_template('live_detection.html')


if __name__ == '__main__':
    app.run(debug=True)
