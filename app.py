import streamlit as st
from PIL import Image
import numpy as np
import joblib
import os

# ====================================================================
# CLASS NAMES
# ====================================================================
CLASS_NAMES = [
    'Normal cases',
    'Benign cases',
    'Malignant cases'
]

# ====================================================================
# TITLE
# ====================================================================
st.title("Lung Cancer Classification")
st.write("Upload a lung image for prediction")

# ====================================================================
# LOAD MODEL
# ====================================================================
model_path = 'final_pipeline_lung_cancer.joblib'

if not os.path.exists(model_path):
    st.error("Model file not found")
    st.stop()

model = joblib.load(model_path)

# แสดงจำนวน feature ที่โมเดลต้องการ (debug)
st.write("Model expects features:", model.n_features_in_)

# ====================================================================
# FILE UPLOADER
# ====================================================================
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

# ====================================================================
# PREDICTION
# ====================================================================
if uploaded_file is not None:

    try:
        # ------------------------------------------------------------
        # LOAD IMAGE
        # ------------------------------------------------------------
        img = Image.open(uploaded_file)

        st.image(img, caption="Uploaded Image", use_container_width=True)

        # ------------------------------------------------------------
        # PREPROCESS (FIXED TO 784 FEATURES)
        # ------------------------------------------------------------

        # grayscale
        img = img.convert('L')

        # resize to 28x28
        img = img.resize((28, 28))

        # to numpy
        img_array = np.array(img)

        # normalize
        img_array = img_array.astype("float32") / 255.0

        # flatten -> 784
        img_array = img_array.reshape(1, 784)

        # ------------------------------------------------------------
        # CHECK FEATURE MATCH
        # ------------------------------------------------------------
        if img_array.shape[1] != model.n_features_in_:
            st.error(
                f"Feature mismatch! Model expects {model.n_features_in_}, "
                f"but got {img_array.shape[1]}"
            )
            st.stop()

        # ------------------------------------------------------------
        # PREDICT
        # ------------------------------------------------------------
        prediction = model.predict(img_array)[0]
        predicted_label = CLASS_NAMES[prediction]

        # ------------------------------------------------------------
        # RESULT
        # ------------------------------------------------------------
        st.success(f"Prediction Result: {predicted_label}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
