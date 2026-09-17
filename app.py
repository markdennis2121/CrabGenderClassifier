import os
import sys
import time
import importlib
import threading
import gc
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Reconfigure stdout/stderr encoding for Windows console compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Reduce TensorFlow / C++ logging noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Ensure PyTorch is set as Keras backend if standalone Keras is used
os.environ['KERAS_BACKEND'] = 'torch'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

# Create uploads directory if not present
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = 'CrabClassifier_v2.keras'

# Global state for lazy model loading
loaded_model = None
model_error = None
preprocess_input_fn = None
load_model_fn = None
model_lock = threading.Lock()


def get_model():
    """Lazily load the model on demand to allow instant server startup on memory-constrained platforms like Render."""
    global loaded_model, model_error, preprocess_input_fn, load_model_fn

    if loaded_model is not None:
        return loaded_model, None, preprocess_input_fn

    if model_error is not None:
        return None, model_error, None

    with model_lock:
        if loaded_model is not None:
            return loaded_model, None, preprocess_input_fn

        if not os.path.exists(MODEL_PATH):
            model_error = f"Model file '{MODEL_PATH}' not found in project directory."
            print(f"[ERROR] {model_error}")
            return None, model_error, None

        print(f"[INFO] Lazy loading model from '{MODEL_PATH}'...")
        start_time = time.time()

        # Attempt loading model via keras or tensorflow.keras dynamically
        try:
            import keras
            preprocess_input_fn = keras.applications.resnet50.preprocess_input
            load_model_fn = keras.models.load_model
        except ImportError:
            try:
                tf_resnet = importlib.import_module('tensorflow.keras.applications.resnet50')
                tf_models = importlib.import_module('tensorflow.keras.models')
                preprocess_input_fn = tf_resnet.preprocess_input
                load_model_fn = tf_models.load_model
            except Exception as e_tf:
                load_model_fn = None
                preprocess_input_fn = None
                model_error = f"Framework import error: {e_tf}"
                print(f"[ERROR] {model_error}")
                return None, model_error, None

        try:
            loaded_model = load_model_fn(MODEL_PATH)
            elapsed = time.time() - start_time
            print(f"[SUCCESS] Successfully loaded Keras model in {elapsed:.2f}s from '{MODEL_PATH}'")
            gc.collect()
            return loaded_model, None, preprocess_input_fn
        except Exception as e:
            model_error = f"Failed to load model file: {e}"
            print(f"[ERROR] {model_error}")
            return None, model_error, None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    model_available = os.path.exists(MODEL_PATH) or (loaded_model is not None)
    return render_template(
        'index.html',
        model_loaded=model_available,
        model_error=model_error if model_error else ("Model file missing" if not model_available else None)
    )


@app.route('/predict', methods=['POST'])
def predict():
    model, err, preprocess_fn = get_model()
    if model is None:
        return jsonify({'error': f'Model is not loaded. Details: {err or "Unknown error"}'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Please upload JPG, PNG, WEBP, or BMP.'}), 400

    try:
        filename = secure_filename(file.filename)
        timestamp = int(time.time() * 1000)
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # 1. Load image & convert to RGB
        img = Image.open(filepath).convert('RGB')
        
        # 2. Resize to 224x224
        img_resized = img.resize((224, 224))
        
        # Convert to float32 numpy array
        img_array = np.array(img_resized, dtype=np.float32)
        
        # Expand dimensions for model batch input (1, 224, 224, 3)
        img_batch = np.expand_dims(img_array, axis=0)

        # 3. ResNet50 preprocess_input
        if preprocess_fn is not None:
            img_preprocessed = preprocess_fn(img_batch)
        else:
            # Fallback zero-center BGR mean subtraction for ResNet50
            img_preprocessed = img_batch[:, :, :, ::-1].copy()
            img_preprocessed[:, :, :, 0] -= 103.939
            img_preprocessed[:, :, :, 1] -= 116.779
            img_preprocessed[:, :, :, 2] -= 123.68

        # 4. Model Prediction
        preds = model.predict(img_preprocessed)
        
        # Format prediction
        if hasattr(preds, 'numpy'):
            preds = preds.numpy()

        if preds.ndim == 2 and preds.shape[1] == 1:
            # Sigmoid activation (0 = Female, 1 = Male)
            prob_male = float(preds[0][0])
            prob_female = 1.0 - prob_male
        elif preds.ndim == 2 and preds.shape[1] == 2:
            # Softmax activation [prob_female, prob_male] (Alphabetical order: F before M)
            prob_female = float(preds[0][0])
            prob_male = float(preds[0][1])
        else:
            flat = preds.flatten()
            if len(flat) >= 2:
                prob_female = float(flat[0])
                prob_male = float(flat[1])
            else:
                prob_male = float(flat[0])
                prob_female = 1.0 - prob_male

        female_confidence = round(prob_female * 100, 2)
        male_confidence = round(prob_male * 100, 2)

        predicted_gender = "Female" if prob_female > prob_male else "Male"

        # Explicit garbage collection to release temporary memory
        gc.collect()

        return jsonify({
            'success': True,
            'predicted_gender': predicted_gender,
            'female_confidence': female_confidence,
            'male_confidence': male_confidence,
            'image_url': f"/static/uploads/{unique_filename}"
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        gc.collect()
        return jsonify({'error': f'Prediction execution error: {str(e)}'}), 500


if __name__ == '__main__':
    # Pre-warm model in development mode
    print("[INFO] Starting Crab Gender Classifier App on http://127.0.0.1:5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)

