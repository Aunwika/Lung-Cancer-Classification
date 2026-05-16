
import streamlit as st
from PIL import Image
import numpy as np
import joblib
import os

# ====================================================================
# TITLE
# ====================================================================

st.title("Lung Cancer Prediction")
st.write("Upload an image for prediction")

# ====================================================================
# LOAD JOBLIB MODEL
# ====================================================================

model_path = 'final_pipeline_lung_cancer.joblib'

if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found")
    st.stop()

# load sklearn pipeline
model = joblib.load(model_path)

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

        # Load image
        img = Image.open(uploaded_file)

        st.image(
            img,
            caption='Uploaded Image',
            use_container_width=True
        )

        st.write("Classifying...")

        # Convert grayscale
        img = img.convert('L')

        # Resize
        img = img.resize((28, 28))

        # Convert to numpy
        img_array = np.array(img)

        # Normalize
        img_array = img_array.astype("float32") / 255.0

        # Flatten image for SVM
        img_array = img_array.flatten().reshape(1, -1)

        # Predict
        prediction = model.predict(img_array)

        # Show result
        st.success(
            f"Prediction: {prediction[0]}"
        )

    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )
