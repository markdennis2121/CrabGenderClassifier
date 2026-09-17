import os
import sys
import time
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

loaded_model = None
model_error = None
preprocess_input_fn = None

# Attempt loading model via keras or tensorflow.keras
try:
    import keras
    from keras.applications.resnet50 import preprocess_input as keras_preprocess
    from keras.models import load_model as keras_load_model
    preprocess_input_fn = keras_preprocess
    load_model_fn = keras_load_model
except Exception as e_keras:
    try:
        import tensorflow as tf
        from tensorflow.keras.applications.resnet50 import preprocess_input as tf_preprocess
        from tensorflow.keras.models import load_model as tf_load_model
        preprocess_input_fn = tf_preprocess
        load_model_fn = tf_load_model
    except Exception as e_tf:
        load_model_fn = None
        preprocess_input_fn = None
        model_error = f"Framework import error: {e_keras}"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

# Create uploads directory if not present
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = 'CrabClassifier_v2.keras'

if os.path.exists(MODEL_PATH):
    if load_model_fn is not None:
        try:
            loaded_model = load_model_fn(MODEL_PATH)
            print(f"[SUCCESS] Successfully loaded Keras model from '{MODEL_PATH}'")
        except Exception as e:
            model_error = f"Failed to load model file: {e}"
            print(f"[ERROR] {model_error}")
    else:
        model_error = "No valid Keras / TensorFlow framework found to load model."
else:
    model_error = f"Model file '{MODEL_PATH}' not found in project directory."
    print(f"[ERROR] {model_error}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html', model_loaded=(loaded_model is not None), model_error=model_error)


@app.route('/predict', methods=['POST'])
def predict():
    if loaded_model is None:
        return jsonify({'error': f'Model is not loaded. Details: {model_error or "Unknown error"}'}), 500

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
        if preprocess_input_fn is not None:
            img_preprocessed = preprocess_input_fn(img_batch)
        else:
            # Fallback zero-center BGR mean subtraction for ResNet50
            img_preprocessed = img_batch[:, :, :, ::-1].copy()
            img_preprocessed[:, :, :, 0] -= 103.939
            img_preprocessed[:, :, :, 1] -= 116.779
            img_preprocessed[:, :, :, 2] -= 123.68

        # 4. Model Prediction
        preds = loaded_model.predict(img_preprocessed)
        
        # Format prediction
        # Output shape could be (1, 1) or (1, 2)
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

        return jsonify({
            'success': True,
            'predicted_gender': predicted_gender,
            'female_confidence': female_confidence,
            'male_confidence': male_confidence,
            'image_url': f"/static/uploads/{unique_filename}"
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'Prediction execution error: {str(e)}'}), 500


if __name__ == '__main__':
    print("[INFO] Starting Crab Gender Classifier App on http://127.0.0.1:5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
