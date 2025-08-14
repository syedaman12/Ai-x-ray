from flask import Flask, render_template, request, send_file
import os
import cv2
from ultralytics import YOLO
import supervision as sv
import pyresearch

# Initialize Flask app
app = Flask(__name__)

# Load YOLO model
model = YOLO("newmodel.pt")

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image processing function
def process_image(input_image_path: str, output_image_path: str):
    # Read the image
    image = cv2.imread(input_image_path)
    if image is None:
        print("Error: Unable to read the image.")
        return

    # Resize the image
    resized = cv2.resize(image, (640, 640))

    # Perform detection
    detections = sv.Detections.from_ultralytics(model(resized)[0])

    # Annotate the image
    annotated = sv.RoundBoxAnnotator().annotate(scene=resized, detections=detections)
    annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=detections)

    # Save the annotated image
    cv2.imwrite(output_image_path, annotated)
    print(f"Processed and saved: {output_image_path}")

# Route to handle image upload
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        
        # If no file is selected
        if file.filename == '':
            return 'No selected file'
        
        # If file is allowed
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Define output image path
            output_filename = os.path.join(app.config['OUTPUT_FOLDER'], 'annotated_' + file.filename)

            # Process the image
            process_image(filename, output_filename)

            # Return the processed image
            return send_file(output_filename, mimetype='image/jpeg')
    
    return render_template('index.html')

if __name__ == "__main__":
    # Create upload and output folders if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    app.run(debug=True)