

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

st.write(
    "Upload a lung image for prediction"
)

# ====================================================================
# LOAD MODEL
# ====================================================================

model_path = 'final_pipeline_lung_cancer.joblib'

if not os.path.exists(model_path):

    st.error("Model file not found")

    st.stop()

# load pipeline
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

        # ------------------------------------------------------------
        # OPEN IMAGE
        # ------------------------------------------------------------

        img = Image.open(uploaded_file)

        st.image(
            img,
            caption="Uploaded Image",
            use_container_width=True
        )

        # ------------------------------------------------------------
        # PREPROCESS
        # ------------------------------------------------------------

        # grayscale
        img = img.convert('L')

        # resize
        img = img.resize((28, 28))

        # convert to numpy
        img_array = np.array(img)

        # normalize
        img_array = img_array.astype("float32") / 255.0

        # flatten
        img_array = img_array.flatten()

        # ------------------------------------------------------------
        # FIX FEATURE SIZE
        # ------------------------------------------------------------

        if len(img_array) < 1164:

            padding = 1164 - len(img_array)

            img_array = np.pad(
                img_array,
                (0, padding)
            )

        # reshape
        img_array = img_array.reshape(1, -1)

        # ------------------------------------------------------------
        # PREDICT
        # ------------------------------------------------------------

        prediction = model.predict(img_array)[0]

        predicted_label = CLASS_NAMES[prediction]

        # ------------------------------------------------------------
        # SHOW RESULT
        # ------------------------------------------------------------

        st.success(
            f"Prediction Result: {predicted_label}"
        )

    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )
