import os
import tensorflow as tf
import numpy as np
import wikipediaapi
from PIL import Image
from flask import Flask, request, render_template, url_for, redirect, jsonify
from werkzeug.utils import secure_filename
import base64
import io

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = os.path.join('models', 'breed_recognition_model_v4.h5')
CLASS_NAMES_PATH = os.path.join('models', 'class_names.txt')
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

IMG_HEIGHT = 224
IMG_WIDTH = 224

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load Model, Class Names, and Wikipedia ---
try:
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Fatal Error: Could not load model. {e}")
    model = None

try:
    print("Loading class names...")
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(class_names)} class names.")
except Exception as e:
    print(f"Fatal Error: Could not load class names. {e}")
    class_names = []

print("Initializing Wikipedia API...")
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='BovineBreedRecognition/1.0 (pradhan.aaryyan@gmail.com)'
)

# --- Core Prediction Function ---
def get_prediction_from_pil_image(pil_image):
    """
    Takes a PIL image, preprocesses it, and returns the predicted breed name.
    """
    try:
        img = pil_image.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        preprocessed_img = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)

        predictions = model.predict(preprocessed_img)
        predicted_index = np.argmax(predictions[0])
        
        if class_names and predicted_index < len(class_names):
            predicted_class_name = class_names[predicted_index]
            return predicted_class_name
        else:
            return "Unknown Breed"

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Prediction Error"

# --- Wikipedia Fetch Function ---
def get_wiki_summary(topic):
    """
    Fetches a summary for a given topic from Wikipedia.
    """
    page = wiki_wiki.page(topic)
    if page.exists():
        # Return the first 3 sentences of the summary
        summary = ". ".join(page.summary.split('. ')[:3]) + "."
        return summary
    return None

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Render the main upload page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    """
    Handle file upload (for web interface) or JSON API request (for orbit-den),
    perform prediction, and render appropriate response.
    """
    if request.is_json:
        # --- Handle JSON API request from orbit-den ---
        data = request.json
        if not data or 'imageBase64' not in data:
            return jsonify({"error": "imageBase64 is required in JSON body"}), 400

        image_b64 = data['imageBase64']
        # Remove data:image/jpeg;base64, prefix if present
        if ';base64,' in image_b64:
            image_b64 = image_b64.split(';base64,')[-1]

        try:
            image_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            return jsonify({"error": f"Failed to decode image: {e}"}), 400

        predicted_breed = get_prediction_from_pil_image(img)
        
        # Return JSON response for API
        return jsonify({"predictions": [{"breed": predicted_breed, "confidence": 1.0}]})

    else:
        # --- Handle traditional web form submission ---
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']

        if file.filename == '' or not file:
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load image from saved file for prediction
            img = Image.open(filepath).convert('RGB')
            predicted_breed = get_prediction_from_pil_image(img)

            # Get Wikipedia summary
            wiki_summary = get_wiki_summary(f"{predicted_breed} (cattle)")
            if not wiki_summary:
                wiki_summary = get_wiki_summary(predicted_breed)

            # The path passed to the template must be relative to the 'static' folder
            image_file_for_template = f'uploads/{filename}'

            return render_template('result.html', 
                                   breed_name=predicted_breed, 
                                   summary=wiki_summary, 
                                   image_file=image_file_for_template)

        return redirect('/')

# --- Main Execution ---
if __name__ == '__main__':
    if model and class_names:
        print("\nSetup complete. Starting Flask server...")
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
    else:
        print("\nCould not start Flask server due to errors during setup.")