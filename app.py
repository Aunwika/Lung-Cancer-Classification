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
# UPLOAD IMAGE
# ====================================================================

uploaded_file = st.file_uploader(
    "Choose Lung Image...",
    type=["jpg", "jpeg", "png"]
)

# ====================================================================
# PREDICTION
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
        # FEATURE EXTRACTION
        # ============================================================

        # grayscale
        gray = img.convert('L')

        # resize
        gray = gray.resize((224, 224))

        # numpy
        gray_np = np.array(gray)

        # normalize
        gray_np = gray_np.astype("float32") / 255.0

        # flatten
        img_array = gray_np.flatten()

        # ============================================================
        # FIX FEATURE SIZE = 1164
        # ============================================================

        if len(img_array) < FEATURE_SIZE:

            padding = FEATURE_SIZE - len(img_array)

            img_array = np.pad(
                img_array,
                (0, padding)
            )

        elif len(img_array) > FEATURE_SIZE:

            img_array = img_array[:FEATURE_SIZE]

        # reshape
        img_array = img_array.reshape(1, -1)

        # show feature size
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
