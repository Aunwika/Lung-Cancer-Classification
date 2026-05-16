%%writefile app.py

import streamlit as st
from PIL import Image
import numpy as np
import joblib
import os

# ====================================================================
# TITLE
# ====================================================================

st.title("MNIST Digit Predictor")

# ====================================================================
# LOAD MODEL
# ====================================================================

model_path = 'final_pipeline_lung_cancer.joblib'

if not os.path.exists(model_path):
    st.error("Model file not found")
    st.stop()

model = joblib.load(model_path)

# ====================================================================
# UPLOAD
# ====================================================================

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

# ====================================================================
# PREDICT
# ====================================================================

if uploaded_file is not None:

    try:

        img = Image.open(uploaded_file)

        st.image(
            img,
            caption="Uploaded Image",
            use_container_width=True
        )

        # grayscale
        img = img.convert('L')

        # resize
        img = img.resize((28, 28))

        # numpy
        img_array = np.array(img)

        # normalize
        img_array = img_array.astype("float32") / 255.0

        # flatten
        img_array = img_array.flatten()

        # ============================================================
        # FIX FEATURE SIZE
        # ============================================================

        if len(img_array) < 1164:

            padding = 1164 - len(img_array)

            img_array = np.pad(
                img_array,
                (0, padding)
            )

        # reshape
        img_array = img_array.reshape(1, -1)

        # predict
        prediction = model.predict(img_array)

        st.success(
            f"Predicted Digit: {prediction[0]}"
        )

    except Exception as e:

        st.error(f"Prediction error: {e}")
