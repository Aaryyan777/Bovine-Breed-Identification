import os
import tensorflow as tf
import numpy as np
import wikipediaapi
from PIL import Image
from flask import Flask, request, render_template, url_for, redirect
from werkzeug.utils import secure_filename

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
def get_prediction(image_path):
    """
    Takes an image path, preprocesses it, and returns the predicted breed name.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
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
    """Handle file upload, prediction, and render result page."""
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    if file.filename == '' or not file:
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get prediction
        predicted_breed = get_prediction(filepath)

        # Get Wikipedia summary
        # We add "(cattle)" to the search query for better specificity
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
        app.run(debug=True)
    else:
        print("\nCould not start Flask server due to errors during setup.")