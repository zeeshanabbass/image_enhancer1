from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageColor
from rembg import remove
import RRDBNet_arch as arch

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'models/RRDB_ESRGAN_x4.pth'
DEVICE = torch.device('cpu')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure upload and processed directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return redirect(url_for('display_image', filename=filename))
    
    return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    processed_filename = request.args.get('processed_filename', None)
    return render_template('index.html', filename=filename, processed_filename=processed_filename)

@app.route('/process', methods=['POST'])
def process_image():
    filename = request.form.get('filename')
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    processed_filename = f"processed_{filename}"
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
    
    # Open the image file
    image = Image.open(input_path).convert('RGBA')

    # Process image based on the form inputs
    try:
        # Remove background (first button)
        if request.form.get('remove_bg') == '1':
            image = remove_background(image)
        
        # Remove background with white default (second button)
        if request.form.get('remove_bg_2') == '1':
            image = remove_background_with_white(image)
        
        # Apply background color change if specified
        bgcolor = request.form.get('bgcolor')
        if bgcolor:
            image = change_background_color(image, bgcolor)
        
        # Crop to passport size if the checkbox is selected
        if 'crop' in request.form:
            image = crop_to_passport_size(image)
        
        # Enhance image quality using the RRDBNet model
        if 'increase-quality' in request.form:
            image = enhance_image_quality(image, MODEL_PATH, DEVICE)
        
        # Convert processed image to RGB if it's in RGBA mode
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Save the processed image
        image.save(output_path, 'JPEG')
    except Exception as e:
        print(f"Error processing image: {e}")
        return redirect(url_for('display_image', filename=filename, processed_filename=None))
    
    return redirect(url_for('display_image', filename=filename, processed_filename=processed_filename))

def remove_background(image):
    # Remove background using rembg
    output = remove(np.array(image))
    return Image.fromarray(output, 'RGBA')

def remove_background_with_white(image):
    # Remove background and set white default background
    output = remove(np.array(image))
    processed_image = Image.fromarray(output, 'RGBA')
    img = np.array(processed_image)
    alpha_channel = img[:, :, 3]  # Alpha channel

    white_bg = np.full_like(img, (255, 255, 255, 255), dtype=np.uint8)
    mask = alpha_channel == 0
    img[mask] = white_bg[mask]

    return Image.fromarray(img, 'RGBA')

def change_background_color(image, color):
    # Change background color
    try:
        bg_color = ImageColor.getrgb(color) + (255,)  # Convert hex to RGBA
    except ValueError:
        bg_color = (0, 0, 255, 255)  # Default to blue if there's an error

    img = np.array(image)
    alpha_channel = img[:, :, 3]  # Alpha channel

    # Create a new background with the specified color
    colored_bg = np.full_like(img, bg_color, dtype=np.uint8)
    
    # Replace transparent areas with the new background color
    mask = alpha_channel == 0
    img[mask] = colored_bg[mask]
    
    return Image.fromarray(img, 'RGBA')

def crop_to_passport_size(image):
    # Crop to passport size
    passport_width, passport_height = 600, 600
    cv_image = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return image

    x, y, w, h = faces[0]
    center_x, center_y = x + w // 2, y + h // 2
    crop_size = max(w, h) * 2
    x1, y1 = max(center_x - crop_size // 2, 0), max(center_y - crop_size // 2, 0)
    x2, y2 = min(center_x + crop_size // 2, cv_image.shape[1]), min(center_y + crop_size // 2, cv_image.shape[0])

    cropped_img = cv_image[y1:y2, x1:x2]
    passport_img = cv2.resize(cropped_img, (passport_width, passport_height), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(cv2.cvtColor(passport_img, cv2.COLOR_BGR2RGB))

def enhance_image_quality(image, model_path, device):
    # Enhance image quality using the RRDBNet model
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    # Preprocess image
    img = np.array(image)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)

    # Enhance image
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    output_img = Image.fromarray(output.astype(np.uint8))
    return output_img

if __name__ == '__main__':
    app.run(debug=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
