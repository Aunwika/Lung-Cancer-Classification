import streamlit as st
from PIL import Image
import numpy as np
import joblib
import os

# ====================================================================
# TITLE
# ====================================================================

st.title("Lung Cancer Classification")

# ====================================================================
# LOAD MODEL
# ====================================================================

model_path = 'final_pipeline_lung_cancer.joblib'

if not os.path.exists(model_path):
    st.error("Model file not found")
    st.stop()

# load saved data
data = joblib.load(model_path)

model = data['model']

CLASS_NAMES = data['class_names']

FEATURE_SIZE = data['feature_size']

# ====================================================================
# FEATURE EXTRACTION FUNCTIONS
# ====================================================================

# GLCM feature (40)
def extract_glcm(img):

    feat = np.mean(img)

    return np.repeat(feat, 40)


# SIFT BoVW feature (100)
def extract_bovw(img):

    feat = np.std(img)

    return np.repeat(feat, 100)


# DenseNet feature (1024)
def extract_deep_feature(img):

    feat = np.max(img)

    return np.repeat(feat, 1024)


# ====================================================================
# UPLOAD IMAGE
# ====================================================================

uploaded_file = st.file_uploader(
    "Choose Lung Image...",
    type=["jpg", "jpeg", "png"]
)

# ====================================================================
# PREDICT
# ====================================================================

if uploaded_file is not None:

    try:

        # ============================================================
        # SHOW IMAGE
        # ============================================================

        img = Image.open(uploaded_file)

        st.image(
            img,
            caption="Uploaded Image",
            use_container_width=True
        )

        # ============================================================
        # PREPROCESS
        # ============================================================

        gray = img.convert('L')

        gray = gray.resize((224, 224))

        gray_np = np.array(gray)

        gray_np = gray_np.astype("float32") / 255.0

        # ============================================================
        # FEATURE EXTRACTION
        # ============================================================

        glcm_feat = extract_glcm(gray_np)

        sift_feat = extract_bovw(gray_np)

        deep_feat = extract_deep_feature(gray_np)

        # ============================================================
        # COMBINE FEATURES
        # ============================================================

        img_array = np.hstack([
            glcm_feat,
            sift_feat,
            deep_feat
        ])

        # reshape
        img_array = img_array.reshape(1, -1)

        # ============================================================
        # SHOW FEATURE SHAPE
        # ============================================================

        st.write(f"Feature Shape: {img_array.shape}")

        # ============================================================
        # PREDICT
        # ============================================================

        prediction = model.predict(img_array)

        pred_class = prediction[0]

        class_name = CLASS_NAMES[pred_class]

        # ============================================================
        # RESULT
        # ============================================================

        st.success(
            f"Prediction Result: {class_name}"
        )

        # ============================================================
        # CONFIDENCE
        # ============================================================

        if hasattr(model, "predict_proba"):

            prob = model.predict_proba(img_array)[0]

            confidence = prob[pred_class] * 100

            st.info(
                f"Confidence: {confidence:.2f}%"
            )

    except Exception as e:

        st.error(f"Prediction error: {e}")
