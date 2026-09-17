import os
import sys
import time
import gc
import numpy as np
from PIL import Image
import streamlit as st

# Environment and Threading Optimizations for Streamlit Cloud (1GB RAM)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KERAS_BACKEND'] = 'torch'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

try:
    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
except Exception:
    pass

# Page Configuration
st.set_page_config(
    page_title="Blue Swimming Crab Gender Classifier | Thesis Project",
    page_icon="🦀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS Styling for Premium Academic Interface
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, .font-outfit {
        font-family: 'Outfit', sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        border-radius: 16px;
        padding: 2.5rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    
    .hero-badge {
        background: #38bdf8;
        color: #0f172a;
        font-weight: 700;
        font-size: 0.8rem;
        padding: 4px 14px;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
        margin-bottom: 0.75rem;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }

    .hero-sub {
        color: #94a3b8;
        font-size: 1rem;
        line-height: 1.6;
    }

    /* Metric Cards */
    .result-card-female {
        background: linear-gradient(135deg, #ec4899 0%, #be185d 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 20px -5px rgba(236, 72, 153, 0.4);
        margin-bottom: 1.5rem;
    }

    .result-card-male {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 20px -5px rgba(59, 130, 246, 0.4);
        margin-bottom: 1.5rem;
    }

    .gender-label {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
        margin-bottom: 0.25rem;
    }

    .confidence-badge {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(8px);
        padding: 6px 18px;
        border-radius: 9999px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Tip Card */
    .tip-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        font-size: 0.875rem;
        color: #475569;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Model paths in order of preference
V3_F16_MODEL_PATH = 'CrabClassifier_v3_float16.keras'
V3_MODEL_PATH = 'CrabClassifier_v3_FineTuned.keras'
V2_MODEL_PATH = 'CrabClassifier_v2.keras'

def is_lfs_pointer(filepath):
    if not os.path.exists(filepath):
        return False
    if os.path.getsize(filepath) < 2048:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(500)
                if 'version https://git-lfs.github.com' in content or 'oid sha256:' in content:
                    return True
        except Exception:
            pass
    return False

def resolve_model_path():
    if os.path.exists(V3_F16_MODEL_PATH) and not is_lfs_pointer(V3_F16_MODEL_PATH):
        return V3_F16_MODEL_PATH
    if os.path.exists(V3_MODEL_PATH) and not is_lfs_pointer(V3_MODEL_PATH):
        return V3_MODEL_PATH
    if os.path.exists(V2_MODEL_PATH) and not is_lfs_pointer(V2_MODEL_PATH):
        return V2_MODEL_PATH
    return V3_F16_MODEL_PATH

# Cached Model Loader for Streamlit
@st.cache_resource(show_spinner="Loading Deep Learning Model...")
def load_cached_model():
    model_path = resolve_model_path()
    if not os.path.exists(model_path) or is_lfs_pointer(model_path):
        return None, f"Model file '{model_path}' is missing or an unpulled LFS pointer.", None

    try:
        import keras
        preprocess_fn = keras.applications.resnet50.preprocess_input
        try:
            model = keras.models.load_model(model_path, compile=False)
        except TypeError:
            model = keras.models.load_model(model_path)
        gc.collect()
        return model, None, preprocess_fn, os.path.basename(model_path)
    except Exception as e:
        return None, str(e), None, None

# Load Model
model, model_err, preprocess_input_fn, active_model_name = load_cached_model()

# Hero Header Banner
st.markdown(f"""
<div class="hero-banner">
    <span class="hero-badge">🎓 Academic Research Demonstration</span>
    <div class="hero-title">Blue Swimming Crab (<i>Portunus pelagicus</i>) Gender Classifier</div>
    <div class="hero-sub">
        Automated Gender Inspection powered by Deep Learning ResNet50 Neural Network (<b>{active_model_name if active_model_name else 'ResNet50'}</b>).
    </div>
</div>
""", unsafe_allow_html=True)

if model_err:
    st.error(f"⚠️ Model Initialization Notice: {model_err}")

# Main Layout Columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Select & Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a Blue Swimming Crab photo (JPG, PNG, WEBP, BMP)",
        type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
        help="Upload clear photos of the crab ventral (underside) abdomen view."
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
    else:
        st.info("👆 Please select or drag a crab image above to start prediction.")

    st.markdown("""
    <div class="tip-card">
        <strong>💡 Image Tip for Best Accuracy:</strong><br>
        Upload clear photos showing the crab ventral (underside) abdomen view, as abdomen shape is the primary diagnostic anatomical feature for gender classification.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("2. Prediction Results")
    
    if uploaded_file is not None and model is not None:
        if st.button("🚀 Predict Gender", type="primary", use_container_width=True):
            with st.spinner("Analyzing anatomical features with ResNet50..."):
                try:
                    # Preprocess Image
                    img_resized = image.resize((224, 224))
                    img_array = np.array(img_resized, dtype=np.float32)
                    img_batch = np.expand_dims(img_array, axis=0)

                    if preprocess_input_fn is not None:
                        img_preprocessed = preprocess_input_fn(img_batch)
                    else:
                        img_preprocessed = img_batch[:, :, :, ::-1].copy()
                        img_preprocessed[:, :, :, 0] -= 103.939
                        img_preprocessed[:, :, :, 1] -= 116.779
                        img_preprocessed[:, :, :, 2] -= 123.68

                    # Inference with PyTorch inference_mode
                    try:
                        import torch
                        with torch.inference_mode():
                            preds = model.predict(img_preprocessed)
                    except Exception:
                        preds = model.predict(img_preprocessed)

                    if hasattr(preds, 'numpy'):
                        preds = preds.numpy()

                    # Format probabilities
                    if preds.ndim == 2 and preds.shape[1] == 1:
                        prob_male = float(preds[0][0])
                        prob_female = 1.0 - prob_male
                    elif preds.ndim == 2 and preds.shape[1] == 2:
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

                    female_conf = round(prob_female * 100, 2)
                    male_conf = round(prob_male * 100, 2)
                    predicted_gender = "Female" if prob_female > prob_male else "Male"
                    top_conf = female_conf if predicted_gender == "Female" else male_conf

                    gc.collect()

                    # Display Dynamic Primary Card
                    card_class = "result-card-female" if predicted_gender == "Female" else "result-card-male"
                    icon_symbol = "♀" if predicted_gender == "Female" else "♂"
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                        <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.85;">Predicted Gender</div>
                        <div class="gender-label">{icon_symbol} {predicted_gender.upper()}</div>
                        <div class="confidence-badge">Primary Classification Confidence: {top_conf:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display Confidence Progress Bars
                    st.markdown("##### 📊 Confidence Probability Breakdown")
                    st.write(f"♀ **Female Confidence**: `{female_conf:.2f}%`")
                    st.progress(int(female_conf))

                    st.write(f"♂ **Male Confidence**: `{male_conf:.2f}%`")
                    st.progress(int(male_conf))

                except Exception as ex:
                    st.error(f"Prediction execution error: {ex}")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem; color: #94a3b8;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🦀</div>
            <h5>Awaiting Image Input</h5>
            <p style="font-size: 0.875rem;">Select or upload a crab image on the left, then click <b>Predict Gender</b> to view confidence analysis.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>Blue Swimming Crab (<i>Portunus pelagicus</i>) Gender Classification Thesis Project • Streamlit Cloud</div>", unsafe_allow_html=True)
